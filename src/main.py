"""Main entry point for the Anima Agent."""

import argparse
import asyncio
import logging
from datetime import datetime
import sys
from pathlib import Path

import structlog

# Configure Python logging level (required for structlog)
logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
)

# Suppress noisy HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
from openai import AsyncOpenAI

from .adapters import ThreadsAdapter
from .agent import AgentBrain, AgentScheduler, Persona
from .agent.scheduler import run_cli_mode
from .memory import AgentMemory
from .observation import (
    ReviewCLI,
    SimulationAnalyzer,
    SimulationLogger,
)
from .observation.report import OnePagerReport
from .threads import MockThreadsClient, ThreadsClient
from .utils import get_settings

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


async def create_agent_brain(
    settings=None, observation_mode: bool = False, use_mock: bool = False
) -> AgentBrain:
    """Create and configure the agent brain.

    Args:
        settings: Application settings (uses get_settings() if None).
        observation_mode: Whether to run in observation mode.
        use_mock: Whether to use mock mode (overrides settings.use_mock_threads).

    Returns:
        Configured AgentBrain instance.
    """
    settings = settings or get_settings()

    # Load persona
    persona_path = Path(settings.persona_file)
    if not persona_path.exists():
        # Try relative to project root
        persona_path = Path(__file__).parent.parent / settings.persona_file

    if not persona_path.exists():
        raise FileNotFoundError(f"Persona file not found: {settings.persona_file}")

    persona = Persona.from_file(persona_path)
    logger.info("persona_loaded", name=persona.identity.name)

    # Initialize OpenAI client
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    # Initialize memory
    # Use separate collection for mock mode to avoid polluting real memories
    is_mock_mode = use_mock or settings.use_mock_threads
    memory_agent_id = settings.agent_name
    if is_mock_mode:
        memory_agent_id = f"{settings.agent_name}_test"
        logger.info("using_test_memory_collection", collection=f"anima_{memory_agent_id}")

    memory = AgentMemory(
        agent_id=memory_agent_id,
        openai_api_key=settings.openai_api_key,
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        database_url=settings.database_url,
        llm_model=settings.openai_model,
    )

    # Initialize simulation logger
    simulation_logger = None
    if observation_mode:
        simulation_logger = SimulationLogger(settings.simulation_data_dir)
        simulation_logger.start_session(
            persona_name=persona.identity.name,
            persona_file=settings.persona_file,
        )
    else:
        # Log real runs separately to avoid mixing with simulation data
        real_log_dir = Path("data/real_logs")
        simulation_logger = SimulationLogger(real_log_dir)
        simulation_logger.start_session(
            persona_name=persona.identity.name,
            persona_file=settings.persona_file,
        )

    # Create brain
    brain = AgentBrain(
        persona=persona,
        platform=None,  # set later when client context is active
        memory=memory,
        openai_client=openai_client,
        model=settings.openai_model,
        advanced_model=settings.openai_model_advanced,
        max_completion_tokens=settings.max_completion_tokens,
        reasoning_effort=settings.reasoning_effort,
        observation_mode=observation_mode,
        simulation_logger=simulation_logger,
    )

    return brain


async def run_daemon(brain: AgentBrain) -> None:
    """Run the agent as a daemon with scheduled tasks."""
    scheduler = AgentScheduler(brain)
    scheduler.start()

    logger.info("agent_daemon_started")

    try:
        # Keep running
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        logger.info("shutting_down")
        scheduler.stop()


async def run_observe_mode(args: argparse.Namespace) -> int:
    """Run in observation mode - simulate but don't post.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code.
    """
    settings = get_settings()
    cycles = getattr(args, "cycles", 1)
    use_mock = getattr(args, "mock", False) or settings.use_mock_threads

    logger.info("starting_observation_mode", cycles=cycles, mock=use_mock)

    brain = await create_agent_brain(settings, observation_mode=True, use_mock=use_mock)

    try:
        # Choose client based on mock flag
        if use_mock:
            logger.info("using_mock_threads_client")
            client_class = MockThreadsClient
        else:
            client_class = ThreadsClient

        async with client_class(
            access_token=settings.threads_access_token or "mock_token",
            user_id=settings.threads_user_id or "mock_user",
        ) as threads:
            brain.platform = ThreadsAdapter(threads)

            for i in range(cycles):
                logger.info("observation_cycle", cycle=i + 1, total=cycles)
                results = await brain.run_cycle()

                if brain.simulation_logger:
                    brain.simulation_logger.increment_cycle()

                logger.info(
                    "cycle_results",
                    cycle=i + 1,
                    simulated_responses=len([r for r in results if r.success]),
                )

        # End session
        if brain.simulation_logger:
            session = brain.simulation_logger.end_session()
            if session:
                print()
                print("=" * 60)
                print("  Observation Mode completed")
                print("=" * 60)
                print(f"  Session ID: {session.id}")
                print(f"  Cycles: {session.cycles_completed}")
                print(f"  Observations: {session.total_observations}")
                print(f"  Decisions: {session.total_decisions}")
                print(f"  Responses: {session.total_responses}")
                print(f"  Reflections: {session.total_reflections}")
                print("=" * 60)
                print()
                print(f"Data saved to: {settings.simulation_data_dir}")
                print("Run 'anima review' to label results")

        return 0

    except Exception as e:
        logger.exception("observation_mode_error", error=str(e))
        return 1
    finally:
        try:
            await brain.close()
        except Exception:
            logger.debug("brain_close_failed", exc_info=True)


def run_review_mode(args: argparse.Namespace) -> int:
    """Run the review/labeling interface.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code.
    """
    settings = get_settings()
    reviewer = ReviewCLI(settings.simulation_data_dir)

    if getattr(args, "stats", False):
        reviewer.show_stats()
        return 0

    if getattr(args, "export", False):
        output = getattr(args, "output", None)
        reviewer.export_labeled_data(output)
        return 0

    response_id = getattr(args, "response_id", None)
    if response_id:
        reviewer.show_response(response_id)
        return 0

    # Default: start interactive review
    limit = getattr(args, "limit", None)
    reviewer.start_review(limit=limit)
    return 0


def run_analyze_mode(args: argparse.Namespace) -> int:
    """Run the analysis and generate suggestions.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code.
    """
    settings = get_settings()
    analyzer = SimulationAnalyzer(settings.simulation_data_dir)

    output = getattr(args, "output", None)
    if output:
        analyzer.export_report(output)
    else:
        analyzer.print_report()

    return 0


def run_report_mode(args: argparse.Namespace) -> int:
    """Generate one-pager report for real/simulation data."""
    settings = get_settings()

    source_dir = getattr(args, "source", None) or settings.simulation_data_dir
    persona_path = getattr(args, "persona", None) or settings.persona_file

    since = None
    if getattr(args, "since", None):
        try:
            since = datetime.fromisoformat(args.since)
        except ValueError:
            logger.error("invalid_since_format", value=args.since)
            return 1

    days = getattr(args, "days", 7)
    exclude_mock = not getattr(args, "include_mock", False)
    recent_limit = getattr(args, "recent_limit", 30)
    output_path = getattr(args, "output", None)
    if output_path:
        output_md = Path(output_path)
    else:
        report_dir = Path("data/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        output_md = report_dir / f"onepager-{datetime.now().date()}.md"

    generator = OnePagerReport(
        data_dir=source_dir,
        persona_path=persona_path,
        since=since,
        days=days,
        exclude_mock=exclude_mock,
        recent_limit=recent_limit,
    )

    generator.generate(output_md=output_md, output_html=getattr(args, "html", False))
    return 0


async def run_webhook_server(args: argparse.Namespace) -> int:
    """Run webhook server to receive external notifications."""
    from .webhooks import ApifyWebhookHandler, WebhookServer

    settings = get_settings()
    brain: AgentBrain | None = None

    # Validate configuration before expensive initialization
    if not settings.webhook_enabled:
        logger.warning("webhook_disabled", msg="Set WEBHOOK_ENABLED=true to enable webhook server")
        return 1

    if not settings.apify_enabled:
        logger.error(
            "no_webhook_handlers",
            msg="No webhook handlers registered. Enable Apify webhook via APIFY_ENABLED=true.",
        )
        return 1

    if not settings.apify_api_token:
        logger.error("apify_webhook_no_token", msg="APIFY_API_TOKEN required for webhook mode")
        return 1

    try:
        brain = await create_agent_brain(settings=settings, use_mock=False)

        # Create webhook server
        server = WebhookServer(
            host=settings.webhook_host,
            port=settings.webhook_port,
            webhook_secret=settings.webhook_secret or None,
        )

        # Setup Apify webhook handler
        apify_handler = ApifyWebhookHandler(
            brain=brain,
            self_username=settings.threads_username,
            max_age_hours=settings.apify_max_age_hours,
            max_items=settings.apify_max_items,
            apify_api_token=settings.apify_api_token,
            max_retries=3,
            retry_delay_base=2.0,
        )

        server.register_handler("apify", apify_handler.handle_webhook)
        logger.info("apify_webhook_registered", path="/webhooks/apify")

        # Start server
        logger.info("webhook_server_starting", host=settings.webhook_host, port=settings.webhook_port)
        await server.start()
        return 0
    except KeyboardInterrupt:
        logger.info("webhook_server_stopped")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("webhook_server_failed", error=str(exc), exc_info=True)
        return 1
    finally:
        logger.info("closing_brain_resources")
        if brain:
            try:
                await brain.close()
            except Exception as e:  # noqa: BLE001
                logger.error("brain_close_failed", error=str(e))


async def async_main(args: argparse.Namespace) -> int:
    """Async main function."""
    # Handle observation mode separately
    if args.mode == "observe":
        return await run_observe_mode(args)

    settings = get_settings()

    # Check for --mock flag or settings
    use_mock = getattr(args, "mock", False) or settings.use_mock_threads

    brain = None
    try:
        brain = await create_agent_brain(settings=settings, use_mock=use_mock)

        # Choose client based on mock setting
        client_class = MockThreadsClient if use_mock else ThreadsClient

        async with client_class(
            access_token=settings.threads_access_token or "mock_token",
            user_id=settings.threads_user_id or "mock_user",
        ) as threads:
            # Attach adapter for the lifetime of this context
            brain.platform = ThreadsAdapter(threads)

            if args.mode == "daemon":
                await run_daemon(brain)
            else:
                await run_cli_mode(
                    brain=brain,
                    mode=args.mode,
                    topic=getattr(args, "topic", None),
                )

        return 0

    except FileNotFoundError as e:
        logger.error("configuration_error", error=str(e))
        return 1
    except Exception as e:
        logger.exception("unexpected_error", error=str(e))
        return 1
    finally:
        if brain:
            # End session for real mode logging
            if brain.simulation_logger:
                brain.simulation_logger.end_session()

            # Ensure underlying clients are closed when leaving
            try:
                await brain.close()
            except Exception:
                logger.debug("brain_close_failed", exc_info=True)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Anima - A persona-driven AI agent with persistent memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal mode
  anima cycle              # Run one interaction cycle
  anima post               # Create an original post
  anima post --topic "AI"  # Post about a specific topic
  anima reflect            # Generate a reflection
  anima stats              # Show agent statistics
  anima daemon             # Run as daemon with scheduler

  # Webhook mode (receive external notifications)
  anima webhook            # Start webhook server for Apify/external providers

  # Observation mode (simulate without posting)
  anima observe            # Run one observation cycle
  anima observe --cycles 5 # Run 5 observation cycles
  anima observe --mock     # Use mock data (no API token needed)
  anima observe --mock --cycles 3

  # Labeling and analysis
  anima review             # Start interactive labeling
  anima review --stats     # Show labeling statistics
  anima analyze            # Generate analysis report
  anima analyze --output report.json
  anima report --days 7    # Generate one-pager report
        """,
    )

    parser.add_argument(
        "mode",
        choices=["cycle", "post", "reflect", "stats", "daemon", "observe", "review", "analyze", "report", "webhook"],
        default="cycle",
        nargs="?",
        help="Operation mode (default: cycle)",
    )

    parser.add_argument(
        "--topic",
        type=str,
        help="Topic for original post (only used with 'post' mode)",
    )

    parser.add_argument(
        "--persona",
        type=str,
        help="Path to persona JSON file (overrides PERSONA_FILE env var)",
    )

    # Observe mode arguments
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of cycles to run in observe mode (default: 1)",
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock Threads client (no API token needed)",
    )

    # Review mode arguments
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show labeling statistics (review mode)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of responses to review (review mode)",
    )

    parser.add_argument(
        "--response-id",
        type=str,
        help="Show a specific response (review mode)",
    )

    parser.add_argument(
        "--export",
        action="store_true",
        help="Export labeled data (review mode)",
    )

    # Analyze mode arguments
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for export/analyze",
    )

    # Report mode arguments
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to include in report (default: 7)",
    )
    parser.add_argument(
        "--since",
        type=str,
        help="ISO date/time to start from (overrides --days)",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Data directory for report source (default: data/simulations)",
    )
    parser.add_argument(
        "--include-mock",
        action="store_true",
        help="Include mock_* records in report (default: exclude)",
    )
    parser.add_argument(
        "--recent-limit",
        type=int,
        default=30,
        help="Recent interaction count to show in report (default: 30)",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Also output HTML alongside Markdown (report mode)",
    )

    args = parser.parse_args()

    # Override persona file if provided
    if args.persona:
        import os
        os.environ["PERSONA_FILE"] = args.persona

    # Handle synchronous modes
    if args.mode == "review":
        return run_review_mode(args)

    if args.mode == "analyze":
        return run_analyze_mode(args)

    if args.mode == "report":
        return run_report_mode(args)

    if args.mode == "webhook":
        return asyncio.run(run_webhook_server(args))

    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())

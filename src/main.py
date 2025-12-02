"""Main entry point for the Anima Agent."""

import argparse
import asyncio
import sys
from pathlib import Path

import structlog
from openai import AsyncOpenAI

from .agent import AgentBrain, AgentScheduler, Persona
from .agent.scheduler import run_cli_mode
from .memory import AgentMemory
from .observation import ReviewCLI, SimulationAnalyzer, SimulationLogger
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
    settings=None, observation_mode: bool = False
) -> AgentBrain:
    """Create and configure the agent brain.

    Args:
        settings: Application settings (uses get_settings() if None).
        observation_mode: Whether to run in observation mode.

    Returns:
        Configured AgentBrain instance.
    """
    print("=== [1] create_agent_brain started ===", flush=True)
    settings = settings or get_settings()
    print("=== [2] settings loaded ===", flush=True)

    # Load persona
    persona_path = Path(settings.persona_file)
    if not persona_path.exists():
        # Try relative to project root
        persona_path = Path(__file__).parent.parent / settings.persona_file

    if not persona_path.exists():
        raise FileNotFoundError(f"Persona file not found: {settings.persona_file}")

    persona = Persona.from_file(persona_path)
    print(f"=== [3] persona loaded: {persona.identity.name} ===", flush=True)
    logger.info("persona_loaded", name=persona.identity.name)

    # Initialize OpenAI client
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    print("=== [4] OpenAI client created ===", flush=True)

    # Initialize memory
    print("=== [5] Creating AgentMemory... ===", flush=True)
    memory = AgentMemory(
        agent_id=settings.agent_name,
        openai_api_key=settings.openai_api_key,
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        database_url=settings.database_url,
    )
    print("=== [6] AgentMemory initialized ===", flush=True)

    # Initialize Threads client (real or mock)
    print(f"=== [7] Creating Threads client (mock={settings.use_mock_threads}) ===", flush=True)
    if settings.use_mock_threads:
        logger.info("using_mock_threads_client")
        threads_client = MockThreadsClient(
            access_token=settings.threads_access_token,
            user_id=settings.threads_user_id or "mock_user",
        )
    else:
        threads_client = ThreadsClient(
            access_token=settings.threads_access_token,
            user_id=settings.threads_user_id,
        )
    print("=== [8] Threads client created ===", flush=True)

    # Initialize simulation logger if in observation mode
    simulation_logger = None
    if observation_mode:
        print("=== [9] Creating SimulationLogger ===", flush=True)
        simulation_logger = SimulationLogger(settings.simulation_data_dir)
        simulation_logger.start_session(
            persona_name=persona.identity.name,
            persona_file=settings.persona_file,
        )
        print("=== [10] SimulationLogger started ===", flush=True)

    # Create brain
    print("=== [11] Creating AgentBrain ===", flush=True)
    brain = AgentBrain(
        persona=persona,
        threads_client=threads_client,
        memory=memory,
        openai_client=openai_client,
        model=settings.openai_model,
        advanced_model=settings.openai_model_advanced,
        observation_mode=observation_mode,
        simulation_logger=simulation_logger,
    )
    print("=== [12] AgentBrain created, returning ===", flush=True)

    return brain


async def run_daemon(brain: AgentBrain) -> None:
    """Run the agent as a daemon with scheduled tasks."""
    print("=== [DAEMON] Starting scheduler ===", flush=True)
    scheduler = AgentScheduler(brain)
    scheduler.start()
    print("=== [DAEMON] Scheduler started ===", flush=True)

    logger.info("agent_daemon_started")

    try:
        # Keep running
        print("=== [DAEMON] Entering main loop ===", flush=True)
        while True:
            await asyncio.sleep(60)
            print("=== [DAEMON] Still alive ===", flush=True)
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
    print("=== [0] run_observe_mode started ===", flush=True)
    settings = get_settings()
    cycles = getattr(args, "cycles", 1)
    use_mock = getattr(args, "mock", False) or settings.use_mock_threads

    print(f"=== Observation mode: cycles={cycles}, mock={use_mock} ===", flush=True)
    logger.info("starting_observation_mode", cycles=cycles, mock=use_mock)

    print("=== Creating agent brain... ===", flush=True)
    brain = await create_agent_brain(settings, observation_mode=True)
    print("=== Agent brain created ===", flush=True)

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
            brain.threads = threads

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
                print("  Observation Mode 完成")
                print("=" * 60)
                print(f"  Session ID: {session.id}")
                print(f"  Cycles: {session.cycles_completed}")
                print(f"  Observations: {session.total_observations}")
                print(f"  Decisions: {session.total_decisions}")
                print(f"  Responses: {session.total_responses}")
                print(f"  Reflections: {session.total_reflections}")
                print("=" * 60)
                print()
                print(f"資料儲存於: {settings.simulation_data_dir}")
                print("執行 'threads-agent review' 來標註結果")

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


async def async_main(args: argparse.Namespace) -> int:
    """Async main function."""
    print(f"=== async_main: args.mode={args.mode} ===", flush=True)
    # Handle observation mode separately
    if args.mode == "observe":
        print("=== Entering observe mode ===", flush=True)
        return await run_observe_mode(args)
    print(f"=== NOT observe mode, running {args.mode} ===", flush=True)

    settings = get_settings()

    try:
        print("=== async_main: calling create_agent_brain (no observation_mode) ===", flush=True)
        brain = await create_agent_brain()

        # Choose client based on mock setting
        if settings.use_mock_threads:
            client_class = MockThreadsClient
        else:
            client_class = ThreadsClient

        async with client_class(
            access_token=settings.threads_access_token or "mock_token",
            user_id=settings.threads_user_id or "mock_user",
        ) as threads:
            # Update brain with context-managed client
            brain.threads = threads

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
  # 正常模式
  anima cycle              # Run one interaction cycle
  anima post               # Create an original post
  anima post --topic "AI"  # Post about a specific topic
  anima reflect            # Generate a reflection
  anima stats              # Show agent statistics
  anima daemon             # Run as daemon with scheduler

  # 觀察模式 (模擬但不發文)
  anima observe            # Run one observation cycle
  anima observe --cycles 5 # Run 5 observation cycles
  anima observe --mock     # Use mock data (no API token needed)
  anima observe --mock --cycles 3

  # 標註與分析
  anima review             # Start interactive labeling
  anima review --stats     # Show labeling statistics
  anima analyze            # Generate analysis report
  anima analyze --output report.json
        """,
    )

    parser.add_argument(
        "mode",
        choices=["cycle", "post", "reflect", "stats", "daemon", "observe", "review", "analyze"],
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

    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())

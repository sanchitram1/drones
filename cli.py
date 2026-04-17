import argparse
import os
import sys
from pathlib import Path


TURN_PROTOCOL_CHOICES = (
    "simple",
    "turn_layer",
    "intersection_cube",
    "sphereabout",
)
DEMAND_MODEL_CHOICES = ("time-series", "fixed-count", "csv")


def normalize_choice(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def parse_turn_protocol(value: str) -> str:
    normalized = normalize_choice(value)
    if normalized not in TURN_PROTOCOL_CHOICES:
        choices = ", ".join(TURN_PROTOCOL_CHOICES)
        raise argparse.ArgumentTypeError(
            f"Unsupported turn protocol '{value}'. Choose from: {choices}."
        )
    return normalized


def parse_demand_model(value: str) -> str:
    normalized = normalize_choice(value).replace("_", "-")
    if normalized not in DEMAND_MODEL_CHOICES:
        choices = ", ".join(DEMAND_MODEL_CHOICES)
        raise argparse.ArgumentTypeError(
            f"Unsupported demand model '{value}'. Choose from: {choices}."
        )
    return normalized


def parse_optional_column(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simulate",
        description="Run the San Francisco real-data drone simulation CLI.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Run the real-data simulation from drone_delivery_sf_realdata_visualization.py",
    )
    add_run_arguments(run_parser)
    run_parser.set_defaults(handler=handle_run)

    return parser


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing sf_nodes.csv, sf_edges.csv, sf_census.csv, and sf_restaurants.csv.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for demand generation and simulation.",
    )
    parser.add_argument(
        "--turn-protocol",
        type=parse_turn_protocol,
        default="simple",
        help=(
            "Turning protocol to use. 'simple' and 'turn_layer' are currently wired "
            "to the same behavior in this engine; the others are accepted but will "
            "raise until that implementation lands."
        ),
    )
    parser.add_argument(
        "--orders-csv",
        type=Path,
        help="Path to an orders CSV with request_time_s, origin_node, and dest_node columns.",
    )
    parser.add_argument(
        "--demand-model",
        type=parse_demand_model,
        default="time-series",
        help="Demand generation mode: time-series, fixed-count, or csv.",
    )
    parser.add_argument(
        "--n-orders",
        type=int,
        help="Number of orders for fixed-count demand.",
    )
    parser.add_argument(
        "--sim-duration-min",
        type=float,
        default=120.0,
        help="Simulation horizon in minutes.",
    )
    parser.add_argument(
        "--dt-s",
        type=int,
        default=30,
        help="Demand discretization step in seconds.",
    )
    parser.add_argument(
        "--base-lambda-per-min",
        type=float,
        default=6.0,
        help="Base arrival rate in orders per minute for time-series demand.",
    )
    parser.add_argument(
        "--peak-multiplier",
        type=float,
        default=1.8,
        help="Multiplier applied during the peak demand window.",
    )
    parser.add_argument(
        "--peak-start-min",
        type=float,
        default=45.0,
        help="Peak-window start time in minutes.",
    )
    parser.add_argument(
        "--peak-end-min",
        type=float,
        default=95.0,
        help="Peak-window end time in minutes.",
    )
    parser.add_argument(
        "--origin-weight-col",
        type=parse_optional_column,
        default=None,
        help="Optional restaurant column to weight origin sampling. Use 'none' for uniform sampling.",
    )
    parser.add_argument(
        "--dest-weight-col",
        type=parse_optional_column,
        default="pop_density",
        help="Census column to weight destination sampling. Use 'none' for uniform sampling.",
    )
    parser.add_argument(
        "--dest-jitter-m",
        type=float,
        default=100.0,
        help="Gaussian destination jitter in meters.",
    )
    parser.add_argument(
        "--cruise-speed-ft-s",
        type=float,
        default=35.0,
        help="Cruise speed in feet per second.",
    )
    parser.add_argument(
        "--climb-rate-ft-s",
        type=float,
        default=8.0,
        help="Climb rate in feet per second.",
    )
    parser.add_argument(
        "--descend-rate-ft-s",
        type=float,
        default=7.0,
        help="Descent rate in feet per second.",
    )
    parser.add_argument(
        "--turn-time-s",
        type=float,
        default=4.0,
        help="Time spent executing the turn maneuver, excluding climb and descent.",
    )
    parser.add_argument(
        "--edge-headway-s",
        type=float,
        default=3.0,
        help="Minimum headway on the same corridor and direction.",
    )
    parser.add_argument(
        "--intersection-headway-s",
        type=float,
        default=4.0,
        help="Minimum headway when reserving an intersection turn window.",
    )
    parser.add_argument(
        "--max-altitude-ft",
        type=float,
        default=400.0,
        help="Maximum allowed altitude in feet.",
    )
    parser.add_argument(
        "--building-clearance-ft",
        type=float,
        default=5.0,
        help="Additional building clearance buffer in feet.",
    )
    parser.add_argument(
        "--collision-radius-ft",
        type=float,
        default=10.0,
        help="Near-conflict screening radius in feet.",
    )
    parser.add_argument(
        "--alt-north-ft",
        type=float,
        default=300.0,
        help="Cruise altitude for northbound segments in feet.",
    )
    parser.add_argument(
        "--alt-south-ft",
        type=float,
        default=315.0,
        help="Cruise altitude for southbound segments in feet.",
    )
    parser.add_argument(
        "--alt-east-ft",
        type=float,
        default=330.0,
        help="Cruise altitude for eastbound segments in feet.",
    )
    parser.add_argument(
        "--alt-west-ft",
        type=float,
        default=345.0,
        help="Cruise altitude for westbound segments in feet.",
    )
    parser.add_argument(
        "--alt-turn-ft",
        type=float,
        default=370.0,
        help="Turn-layer altitude in feet.",
    )
    parser.add_argument(
        "--max-demo-routes-3d",
        type=int,
        default=8,
        help="Maximum number of routes shown in the 3D plot.",
    )
    parser.add_argument(
        "--max-demo-routes-map",
        type=int,
        default=120,
        help="Maximum number of routes drawn on the top-down map.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively.",
    )
    parser.add_argument(
        "--save-figure",
        type=Path,
        help="Base path or prefix for saved figures. Writes *_overview.png and *_summary.png.",
    )
    parser.add_argument(
        "--save-results",
        type=Path,
        help="Base path or prefix for saved CSV/JSON outputs.",
    )
    parser.add_argument(
        "--save-orders",
        type=Path,
        help="Write the resolved orders table to this CSV path.",
    )


def handle_run(args: argparse.Namespace) -> int:
    if not args.show:
        os.environ.setdefault("MPLBACKEND", "Agg")

    from drone_delivery_sf_realdata_visualization import SimConfig, run_simulation

    cfg = SimConfig(
        data_dir=str(args.data_dir),
        random_seed=args.seed,
        turn_protocol=args.turn_protocol,
        demand_model=args.demand_model,
        sim_duration_s=int(args.sim_duration_min * 60),
        dt_s=args.dt_s,
        base_lambda_per_min=args.base_lambda_per_min,
        peak_multiplier=args.peak_multiplier,
        peak_window_s=(
            int(args.peak_start_min * 60),
            int(args.peak_end_min * 60),
        ),
        n_orders_override=args.n_orders,
        origin_weight_col=args.origin_weight_col,
        dest_weight_col=args.dest_weight_col,
        dest_jitter_m=args.dest_jitter_m,
        cruise_speed_ft_s=args.cruise_speed_ft_s,
        climb_rate_ft_s=args.climb_rate_ft_s,
        descend_rate_ft_s=args.descend_rate_ft_s,
        turn_time_s=args.turn_time_s,
        max_altitude_ft=args.max_altitude_ft,
        building_clearance_ft=args.building_clearance_ft,
        collision_radius_ft=args.collision_radius_ft,
        alt_north_ft=args.alt_north_ft,
        alt_south_ft=args.alt_south_ft,
        alt_east_ft=args.alt_east_ft,
        alt_west_ft=args.alt_west_ft,
        alt_turn_ft=args.alt_turn_ft,
        edge_time_headway_s=args.edge_headway_s,
        intersection_headway_s=args.intersection_headway_s,
        max_demo_routes_3d=args.max_demo_routes_3d,
        max_demo_routes_map=args.max_demo_routes_map,
    )

    run_simulation(
        cfg=cfg,
        orders_csv=args.orders_csv,
        show=args.show,
        save_figure=args.save_figure,
        save_results=args.save_results,
        save_orders=args.save_orders,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] not in {"run", "-h", "--help"}:
        argv = ["run", *argv]

    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    """Parse command line arguments."""

    parser = ArgumentParser(
        description="Compute and visualize Lagrange points in a system of two masses."
    )

    sim_group = parser.add_argument_group("Simulation parameters")
    sim_group.add_argument(
        "-d",
        "--sep",
        type=float,
        default=1.0,
        help="Separation between the two main bodies",
    )
    sim_group.add_argument(
        "-M",
        "--mass-1",
        type=float,
        default=3.0,
        help="Mass of the larger body",
    )
    sim_group.add_argument(
        "-m",
        "--mass-2",
        type=float,
        default=1.0,
        help="Mass of the smaller body",
    )

    numeric_group = parser.add_argument_group("Numerical solver parameters")
    numeric_group.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Tolerance for the root finding algorithm",
    )
    numeric_group.add_argument(
        "--maxiter",
        type=int,
        default=100,
        help="Maximum number of iterations for the root finding algorithm",
    )

    plot_group = parser.add_argument_group("Plotting parameters")
    plot_group.add_argument(
        "--grid-size",
        type=int,
        default=200,
        help="Number of grid points to use for plotting",
    )
    plot_group.add_argument(
        "--xlim",
        required=False,
        action="store",
        nargs=2,
        help="x-axis limits",
    )
    plot_group.add_argument(
        "--ylim",
        required=False,
        nargs=2,
        help="y-axis limits",
    )

    args = parser.parse_args()

    if args.xlim is None:
        args.xlim = [-1.5 * args.sep, 1.5 * args.sep]
    else:
        args.xlim = [float(x) for x in args.xlim]

    if args.ylim is None:
        args.ylim = [-1.5 * args.sep, 1.5 * args.sep]
    else:
        args.ylim = [float(y) for y in args.ylim]

    return args

import argparse
from pcd_fusion import PCDFusionPublisher


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--spatial_cutoff",
        type=float,
        default=[-1.0, 1.0, -0.5, 0.5, 0.0, 0.3],
        nargs=6,
        required=True,
        help="Spatial cutoff for pointcloud data in [x_min, x_max, y_min, y_max, z_min, z_max].",
    )
    args.add_argument(
        "--downsample_N",
        type=int,
        default=4096,
        help="Number of points to downsample to.",
    )
    args.add_argument(
        "--publish_freq",
        type=int,
        default=30,
        help="Frequency to publish fused pointcloud.",
    )
    args.add_argument(
        "--use_fps",
        action="store_true",
        help="Whether to use fpsample to downsample.",
    )
    args.add_argument(
        "--fps_h",
        type=int,
        default=5,
        help="The `h` parameter for bucket_fps_kdline_sampling",
    )
    args = args.parse_args()
    fusion_publisher = PCDFusionPublisher(
        downsample_N=args.downsample_N,
        publish_freq=args.publish_freq,
        spatial_cutoff=args.spatial_cutoff,
        use_fps=args.use_fps,
        fps_h=args.fps_h,
    )
    fusion_publisher.run()

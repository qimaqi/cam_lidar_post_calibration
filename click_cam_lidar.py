#!/usr/bin/env python3
import click

from cam_lidar import CamLidarTool


@click.command()
@click.argument('data_path')
@click.option('--data_id', '-id', default=0)
@click.option('--extract', '-ex', default=True)


def run_cam_lidar_calib(data_path, data_id, extract):
    print('Cam Lidar calibration called')
    tool = CamLidarTool(data_path, data_id, extract)
    tool.run()


if __name__ == '__main__':
    run_cam_lidar_calib()  # pylint: disable=no-value-for-parameter
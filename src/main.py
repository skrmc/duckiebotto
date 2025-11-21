"""Entry point for the Duckiebot controllers."""

from __future__ import annotations

import os
import threading

import rospy

from rosnodes import DuckiebotNode
from state import SharedState
from web import create_app, run_server


def _start_web(app) -> None:
    run_server(app)


def main() -> None:
    vehicle_name = os.environ.get("VEHICLE_NAME")
    if not vehicle_name:
        raise RuntimeError("VEHICLE_NAME environment variable is required")

    rospy.init_node("duckiebot_manager", anonymous=False)
    state = SharedState(vehicle_name)

    DuckiebotNode(vehicle_name, state)

    app = create_app(state)
    web_thread = threading.Thread(target=_start_web, args=(app,), daemon=True)
    web_thread.start()

    rospy.loginfo("Duckiebot stack started for %s", vehicle_name)

    rospy.spin()


if __name__ == "__main__":
    main()

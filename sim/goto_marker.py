"""Go-to-marker mode (`python main.py --goto`): drive the bot to the ArUco target
marker (shared.TARGET_ID, printed by vision/tools/make_markers.py) using the camera.

No trained policy: each step the bot either turns 45 deg toward the target or
steps 5 cm forward, the same moves the policy uses. It stops next to the marker
(the bot would hide it by driving over it) and, in follow mode, sets off again
when the marker is moved. Distances are in cm, angles in radians, as in sim/.
"""
import time

import numpy as np

from obs import travel_room, wrap

ARRIVE_CM = 18.0  # stop when the bot centre is this close (robot radius 10 + a step + 3)
RESUME_CM = 28.0  # follow mode: set off again once the target is this far away
AIM_DEG = 22.5    # step forward when the target is within this of the heading
                  # (half a turn, so a turn always brings it back inside)


def next_action(pos, theta, target, front_room):
    """
    What to do next: "arrived", "blocked" (the next forward step would leave
    the desk), or a move action: 0 forward, 2 heading +45, 3 heading -45.
    pos and target are (x, y) in cm, theta the heading in rad, front_room the
    cm the bot can still drive forward (obs.travel_room).
    """
    to_target = np.asarray(target, float) - np.asarray(pos, float)
    if np.hypot(*to_target) <= ARRIVE_CM:
        return "arrived"
    off = np.rad2deg(wrap(np.arctan2(to_target[1], to_target[0]) - theta))
    if abs(off) > AIM_DEG:
        return 2 if off > 0 else 3  # +45 makes the heading increase
    return "blocked" if front_room < 5.0 else 0


def goto_marker(hw, log=print, should_stop=None, follow=True, max_moves=None, poll=0.2):
    """
    Drive `hw` (a VisionHardware) to the target marker. Returns
    {"reason", "moves"}: reason is "arrived" (only when follow is False),
    "stopped" (should_stop() turned True) or "max moves".
    In follow mode it keeps running until should_stop() is True.
    """
    should_stop = should_stop or (lambda: False)
    moves, arrived, waiting = 0, False, None

    def wait(reason, message):
        nonlocal waiting
        if waiting != reason:  # log each wait once, not every poll
            log(message)
            waiting = reason
        time.sleep(poll)

    while not should_stop():
        if max_moves is not None and moves >= max_moves:
            return {"reason": "max moves", "moves": moves}
        pose, target = hw.pose(), hw.target()
        if pose is None:
            wait("bot", "Can't see the bot marker: waiting.")
            continue
        if target is None:
            wait("target", "Can't see the target marker (ID 5): waiting.")
            continue
        pos, theta = pose
        distance = float(np.hypot(*(target - pos)))
        if arrived and distance < RESUME_CM:
            time.sleep(poll)  # still at the target
            continue
        W, H = hw.desk_cm()
        action = next_action(pos, theta, target, travel_room(pos, theta, W, H, hw.m.robot_radius))
        if action == "arrived":
            log(f"Reached the target ({distance:.0f} cm away) after {moves} moves."
                + (" Move the marker and the bot will follow." if follow else ""))
            hw.stop()
            if not follow:
                return {"reason": "arrived", "moves": moves}
            arrived, waiting = True, None
            continue
        if action == "blocked":
            wait("blocked", "The target is too close to the desk edge to reach "
                            "without driving off: waiting for it to move.")
            continue
        if arrived or waiting is not None:
            log(f"Driving to the target, {distance:.0f} cm away.")
        arrived, waiting = False, None
        hw.execute(action)
        moves += 1
    return {"reason": "stopped", "moves": moves}

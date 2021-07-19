from datetime import datetime, timezone, timedelta


def now_stamp():
    now = datetime.now(timezone(timedelta(hours=9)))
    yd_string = now.strftime('%Y%m%d')
    time_string = now.strftime('%H%M%S')
    return yd_string, time_string

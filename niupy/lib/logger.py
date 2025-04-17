"""
Logging system

Log level
---------

======= ======
Level   number
------- ------
DEBUG   6
EXTRA   5
INFO    4
NOTE    3
WARN    2
ERROR   1
QUIET   0
======= ======
"""

import sys
import time

if sys.version_info < (3, 0):
    process_clock = time.clock
    perf_counter = time.time
else:
    process_clock = time.process_time
    perf_counter = time.perf_counter

DEBUG = 6
EXTRA = 5
INFO = 4
NOTE = 3
WARN = 2
ERROR = 1
QUIET = 0

DEBUG_LEVEL = 6
TIMER_LEVEL = 5
TIMER0_LEVEL = 4


def flush(rec, msg, *args):
    rec.stdout.write(msg % args)
    rec.stdout.write("\n")
    rec.stdout.flush()


def log(rec, msg, *args):
    if rec.verbose > QUIET:
        flush(rec, msg, *args)


def error(rec, msg, *args):
    if rec.verbose >= ERROR:
        flush(rec, "\nERROR: " + msg + "\n", *args)
    sys.stderr.write("ERROR: " + (msg % args) + "\n")


def warn(rec, msg, *args):
    if rec.verbose >= WARN:
        flush(rec, "\nWARN: " + msg + "\n", *args)
        if rec.stdout is not sys.stdout:
            sys.stderr.write("WARN: " + (msg % args) + "\n")


def info(rec, msg, *args):
    if rec.verbose >= INFO:
        flush(rec, msg, *args)


def note(rec, msg, *args):
    if rec.verbose >= NOTE:
        flush(rec, msg, *args)


def extra(rec, msg, *args):
    if rec.verbose >= EXTRA:
        flush(rec, msg, *args)


def debug(rec, msg, *args):
    if rec.verbose >= DEBUG:
        flush(rec, msg, *args)


def stdout(rec, msg, *args):
    if rec.verbose >= DEBUG:
        flush(rec, msg, *args)
    sys.stdout.write(">>> %s\n" % msg)


def timer(rec, msg, cpu0=None, wall0=None):
    if cpu0 is None:
        cpu0 = rec._t0
    if wall0:
        rec._t0, rec._w0 = process_clock(), perf_counter()
        if rec.verbose >= TIMER_LEVEL:
            flush(
                rec,
                "> CPU time for %s %9.2f sec, wall time %9.2f sec"
                % (msg, rec._t0 - cpu0, rec._w0 - wall0),
            )
        return rec._t0, rec._w0
    else:
        rec._t0 = process_clock()
        if rec.verbose >= TIMER_LEVEL:
            flush(rec, "> CPU time for %s %9.2f sec" % (msg, rec._t0 - cpu0))
        return rec._t0


def timer0(rec, msg, cpu0=None, wall0=None):
    if cpu0 is None:
        cpu0 = rec._t0
    if wall0:
        rec._t0, rec._w0 = process_clock(), perf_counter()
        if rec.verbose >= TIMER0_LEVEL:
            flush(
                rec,
                "> CPU time for %s %9.2f sec, wall time %9.2f sec"
                % (msg, rec._t0 - cpu0, rec._w0 - wall0),
            )
        return rec._t0, rec._w0
    else:
        rec._t0 = process_clock()
        if rec.verbose >= TIMER0_LEVEL:
            flush(rec, "> CPU time for %s %9.2f sec" % (msg, rec._t0 - cpu0))
        return rec._t0


def timer_debug(rec, msg, cpu0=None, wall0=None):
    if cpu0 is None:
        cpu0 = rec._t0
    if wall0:
        rec._t0, rec._w0 = process_clock(), perf_counter()
        if rec.verbose >= DEBUG_LEVEL:
            flush(
                rec,
                "> CPU time for %s %9.2f sec, wall time %9.2f sec"
                % (msg, rec._t0 - cpu0, rec._w0 - wall0),
            )
        return rec._t0, rec._w0
    else:
        rec._t0 = process_clock()
        if rec.verbose >= DEBUG_LEVEL:
            flush(rec, "> CPU time for %s %9.2f sec" % (msg, rec._t0 - cpu0))
        return rec._t0


def niupy_header(self):
    width = 80
    center = lambda text: text.center(width)

    lines = [
        "-" * width,
        center("NiuPy: Open-Source implementation of EOM-DSRG methods"),
        center("Version 0.1"),
        center("Copyright (c) 2025 Shuhang Li"),
        center("Available at https://github.com/shuhangli98/niupy"),
        "-" * width,
    ]

    self.info("\n\n" + "\n\n".join(lines) + "\n")


class Logger:
    """
    Attributes:
        stdout : file object or sys.stdout
            The file to dump output message.
        verbose : int
            Large value means more noise in the output file.
    """

    def __init__(self, stdout=sys.stdout, verbose=NOTE):
        self.stdout = stdout
        self.verbose = verbose
        self._t0 = process_clock()
        self._w0 = perf_counter()

    log = log
    error = error
    warn = warn
    info = info
    note = note
    extra = extra
    debug = debug
    timer0 = timer0
    timer = timer
    timer_debug = timer_debug
    niupy_header = niupy_header


def new_logger(rec=None, verbose=None):
    """Create and return a :class:`Logger` object

    Args:
        rec : An object which carries the attributes stdout and verbose

        verbose : a Logger object, or integer or None
            The verbose level. If verbose is a Logger object, the Logger
            object is returned. If verbose is not specified (None),
            rec.verbose will be used in the new Logger object.
    """
    if isinstance(verbose, Logger):
        log = verbose
    elif isinstance(verbose, int):
        if getattr(rec, "stdout", None):
            log = Logger(rec.stdout, verbose)
        else:
            log = Logger(sys.stdout, verbose)
    else:
        log = Logger(rec.stdout, rec.verbose)
    return log

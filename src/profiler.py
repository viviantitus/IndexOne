import cProfile

def profile_with_args(enabled):

    def profileit(func):
        """
        Decorator (function wrapper) that profiles a single function
        @profileit()
        def func1(...)
                # do something
            pass
        """
        def wrapper(*args, **kwargs):
            if enabled:
                prof = cProfile.Profile()
                retval = prof.runcall(func, *args, **kwargs)
                prof.print_stats()
                return retval
            else:
                return func(*args, **kwargs)

        return wrapper
    return profileit
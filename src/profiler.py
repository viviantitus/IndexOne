import cProfile

def profileit(func):
    """
	Decorator (function wrapper) that profiles a single function
	@profileit()
	def func1(...)
            # do something
	    pass
    """
    def wrapper(*args, **kwargs):
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.print_stats()
        return retval

    return wrapper
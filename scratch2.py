
def fibo(n):
    """
    calculates the nth term in the fibinoci sequence
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    else:
        return fibo(n-1) + fibo(n-2)


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

#------------------ test out mulitple init functions --------------------------
class InitTest:
#    def __init__(self,one,two):
#        self.one = 1
#        self.two = 2
#    def __init__(self,one):
#        two = 2
#        self.__init__(one,two)
    
#note, approach does not work, there doesn't seem to be a good approach to multiple 
#constructors in python besides default arguments
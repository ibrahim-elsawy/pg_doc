class test():
    def __init__(self):
        self.x = 200
    def __call__(self, y):
        print('called ', y)


z = test()
z(300)

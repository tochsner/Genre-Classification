class Fahrzeug:
    def fahren():
        pass

class Auto(Fahrzeug):
    def __init__(self):
        self.v = 100
    def fahren():
        print("Langsam fahren")

class Porsche(Auto):
    def __init__(self):
        self.v = 1000
    def fahren():
        print("Sehr schnell fahren")

print("Schwanzlurche " * 100)
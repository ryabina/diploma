
class p_match:
    def __init__(self, u1p,v1p,u2p,v2p, u1c,v1c,u2c,v2c):
        self.u1p = u1p #координата по вертикали на "предыдущем" левом изображении
        self.v1p = v1p #координата по горизонтали на "предыдущем" левом изображении
        self.u2p = u2p #координата по вертикали на "предыдущем" правом изображении
        self.v2p = v2p
        self.u1c = u1c #координата по вертикали на "текущем" левом изображении
        self.v1c = v1c
        self.u2c = u2c #координата по вертикали на "текущем" правом изображении
        self.v2c = v2c

p = [] # список, хранящий структуру p_match

files = 'tryingOCV_4pict_p2.dat'
f = open(files, 'rb')
print '1'

i = 0
while True:
    try:
        p.append( pickle.load(f))
    except (EOFError):
        break



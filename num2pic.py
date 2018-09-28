from PIL import Image

def createLoadImg(loadNodes=[], loadValues=[], shape_x=180, shape_y=60, path='./test.png'):
    # 先给所有图像都弄成灰色表示无负载
    im = Image.new("RGB", (shape_x, shape_y), "grey")
    loads = {}
    for (node, value) in zip(loadNodes, loadValues):
        n = int(node)
        if n % 2 == 0:
            z = n / 2
            x = int(z / 61)
            y = int(z - x * 61)
            if (x, y) not in loads:
                loads[(x, y)] = [128, 128, 128]
            if value == -1:
                loads[(x, y)][1] = 255
        else:
            z = (n - 1) / 2
            x = int(z / 61)
            y = int(z - x * 61)
            if (x, y) not in loads:
                loads[(x, y)] = [128, 128, 128]
            if value == -1:
                loads[(x, y)][0] = 255
    for key, value in loads.items():
        try:
            im.putpixel(key, tuple(value))
        except BaseException:
            pass
    im.save(path)


def createSupportImg(fixNodesX=[], fixNodesY=[], shape_x=180, shape_y=60, path='./test.png'):
    # 先给所有图像都弄成灰色表示无负载
    im = Image.new("RGB", (shape_x, shape_y), "black")
    fix = {}
    for X in fixNodesX:
        fixX = int(X)
        if fixX % 2 == 0:
            z = fixX / 2
            x = int(z / 61)
            y = int(z - x * 61)
            if (x, y) not in fix:
                fix[(x, y)] = [0, 0, 0]
            fix[(x, y)][1] = 255
        else:
            z = (fixX - 1) / 2
            x = int(z / 61)
            y = int(z - x * 61)
            if (x, y) not in fix:
                fix[(x, y)] = [0, 0, 0]
            fix[(x, y)][1] = 255
    for Y in fixNodesY:
        fixY = int(Y)
        if fixY % 2 == 0:
            z = fixY / 2
            x = int(z / 61)
            y = int(z - x * 61)
            if (x, y) not in fix:
                fix[(x, y)] = [0, 0, 0]
            fix[(x, y)][0] = 255
        else:
            z = (fixY - 1) / 2
            x = int(z / 61)
            y = int(z - x * 61)
            if (x, y) not in fix:
                fix[(x, y)] = [0, 0, 0]
            fix[(x, y)][0] = 255
    for key, value in fix.items():
        try:
            im.putpixel(key, tuple(value))
        except BaseException:
            pass
    im.save(path)




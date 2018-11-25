# -*- coding:utf-8 -*-
"""
道格拉斯算法的实现
程序需要安装shapely模块
"""
import math
from shapely import wkt, geometry
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import logging


class Point(object):
    """点类"""
    x = 0.0
    y = 0.0
    index = 0  # 点在线上的索引

    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.index = index


class Douglas(object):
    """道格拉斯算法类"""
    D = 2  # 容差

    def readPoint(self,line):
        points = []
        """生成点要素"""
        g = wkt.loads(line)
        # g = wkt.loads('LINESTRING(1 4,2 3,4 4,9 4,7 7,8 6,9 10,1 10)')
        coords = g.coords
        # coords = geometry.LineString([(1,4),(2,3),(4,2),(6,6),(7,7),(8,6),(9,5),(10,10)])
        for i in range(len(coords)):
            points.append(Point(coords[i][0], coords[i][1], i))
        return points

    def compress(self, points, p1, p2):
        """具体的抽稀算法"""
        swichvalue = False
        # 一般式直线方程系数 A*x+B*y+C=0,利用点斜式
        # A=(p1.y-p2.y)/math.sqrt(math.pow(p1.y-p2.y,2)+math.pow(p1.x-p2.x,2))
        A = (p1.x - p2.x)
        # B=(p2.x-p1.x)/math.sqrt(math.pow(p1.y-p2.y,2)+math.pow(p1.x-p2.x,2))
        B = (p2.y - p1.y)
        # C=(p1.x*p2.y-p2.x*p1.y)/math.sqrt(math.pow(p1.y-p2.y,2)+math.pow(p1.x-p2.x,2))
        C = (p1.y * p2.x - p2.y * p1.x)

        m = points.index(p1)
        n = points.index(p2)
        distance = []
        middle = None

        if (n == m + 1 or n == m):
            return
        # 计算中间点到直线的距离
        logging.info('m:%d,n:%d'%(m,n))
        for i in range(m + 1, n):
            d = abs(A * points[i].y + B * points[i].x + C) / math.sqrt(math.pow(A, 2) + math.pow(B, 2))
            # array_longi = np.array([B, A])
            # array_trans = np.array([p2.x-points[i].x, p2.y-points[i].y])
            # # 用向量计算点到直线距离
            # array_temp = (float(array_trans.dot(array_longi)) / array_longi.dot(array_longi))  # 注意转成浮点数运算
            # array_temp = array_longi.dot(array_temp)
            # d = np.sqrt((array_trans - array_temp).dot(array_trans - array_temp))
            distance.append(d)

        dmax = max(distance)

        if dmax > self.D:
            swichvalue = True
        else:
            swichvalue = False

        if (not swichvalue):
            for item in points[m+1:(n-m)]:
                points.remove(item)
        else:
            index = distance.index(dmax)
            middle = points[m:][index+1]
            self.compress(points,p1, middle)
            self.compress(points,middle, p2)





def main():
    """测试"""
    # p=Point(20,20,1)
    # print '%d,%d,%d'%(p.x,p.x,p.index)

    d = Douglas()
    points = d.readPoint()
    # d.printPoint()
    # 结果图形的绘制，抽稀之前绘制
    fig = plt.figure()
    a1 = fig.add_subplot(121)
    dx = []
    dy = []
    for i in range(len(points)):
        dx.append(points[i].x)
        dy.append(points[i].y)
    a1.plot(dx, dy, color='g', linestyle='-', marker='+')

    d.compress(points,points[0], points[len(points) - 1])

    # 抽稀之后绘制
    dx1 = []
    dy1 = []
    a2 = fig.add_subplot(122)
    for p in points:
        dx1.append(p.x)
        dy1.append(p.y)
    a2.plot(dx1, dy1, color='r', linestyle='-', marker='+')

    # print "========================\n"
    # d.printPoint()

    plt.show()


def draw_image_cover(inkarray):
    newpoint = []
    for xarr,yarr in inkarray:
        linestring = 'LINESTRING('
        n = 0
        stroke = []
        xpoints = []
        ypoints = []
        if len(xarr) > 2:
            for x in xarr:
                linestring += str(x) + " " + str(yarr[n])+','
                n += 1
            linestring = linestring[0:len(linestring)-1]
            linestring += ')'
            d = Douglas()
            points = d.readPoint(linestring)
            d.compress(points,points[0], points[len(points) - 1])
            for point in points:
                xpoints.append(int(point.x))
                ypoints.append(int(point.y))
        else:
            xpoints = xarr.copy()
            ypoints = yarr.copy()
        stroke.append(tuple(xpoints))
        stroke.append(tuple(ypoints))
        newpoint.append(tuple(stroke))
    print(newpoint)

def show_draw():
    ink = [((66, 113, 130, 132, 151, 153, 155, 157, 159, 160, 162, 163, 165, 166, 174, 174, 175, 176, 176, 177, 178, 179, 179, 180, 181, 181, 182, 183, 184, 186, 187, 189, 190, 192, 194, 196, 198, 200, 202, 205, 207, 209, 211, 213, 216, 218, 220, 222, 224, 225, 227, 229, 230, 232, 232, 234, 235, 235, 236, 237, 238, 238, 239, 239, 240, 241, 242, 242, 243, 244, 245, 245, 245, 245, 246, 246, 246, 247, 247, 247, 247, 247, 247, 247, 247, 246, 245, 244, 242, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232, 232, 231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 217, 216, 213, 211, 209, 207, 205, 204, 203, 201, 200, 197, 195, 192, 190, 189, 187, 186, 185, 183, 182, 180, 177, 175, 171, 169, 166, 163, 161, 157, 153, 150, 142, 139, 136, 130, 128, 126, 124, 121, 120, 116, 114, 112, 110, 108, 107, 104, 101, 98, 97, 96, 95, 94, 93, 91, 89, 88, 86, 85, 83, 81, 80, 79, 78, 77, 75, 74, 72, 72, 71, 70, 70, 69, 69, 69, 68, 68, 67, 67, 66, 66, 66, 65, 65), (94, 67, 63, 63, 62, 62, 62, 62, 62, 63, 63, 64, 66, 67, 82, 83, 85, 87, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 102, 103, 104, 104, 105, 105, 105, 106, 106, 106, 106, 107, 107, 107, 107, 107, 107, 107, 106, 105, 104, 104, 103, 102, 101, 100, 99, 97, 96, 95, 93, 92, 90, 87, 86, 83, 81, 80, 78, 76, 75, 74, 73, 72, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 60, 59, 58, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 56, 56, 56, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 77, 80, 84, 87, 90, 92, 94, 96, 99, 102, 105, 110, 112, 113, 115, 117, 118, 119, 120, 121, 123, 124, 125, 126, 126, 126, 126, 126, 126, 126, 125, 124, 123, 122, 121, 120, 120, 119, 118, 118, 117, 117, 116, 116, 116, 115, 114, 114, 114, 113, 113, 113, 112, 112, 111, 110, 110, 109, 108, 106, 105, 104, 103, 103, 102, 101, 101, 100, 100, 100, 99, 99, 98, 97, 97, 96, 96, 95, 95, 94, 93, 93, 92)), ((100,), (92,)), ((100,), (92,)), ((91, 97, 100, 100, 98, 97, 96, 95, 94, 94, 93, 93, 93), (94, 94, 92, 91, 90, 90, 90, 90, 90, 91, 92, 93, 94)), ((102, 107, 108, 108, 105, 105, 104, 104), (75, 88, 103, 104, 109, 110, 111, 112))]
    image = Image.new("RGB", (255, 255), color=(255, 255, 255))
    image_draw = ImageDraw.Draw(image)
    _strokes = []
    for stroke in ink:
        points = []
        xs = stroke[0]
        ys = stroke[1]

        if len(xs) != len(ys):
            raise Exception("something is wrong, different number of x's and y's")

        for point in range(len(xs)):
            x = xs[point]
            y = ys[point]
            points.append((x, y))
        _strokes.append(points)
    for stroke in _strokes:
        image_draw.line(stroke, fill=(0, 0, 0), width=2)
    # image.save("/app/test.png")
    image.show()


# if __name__ == '__main__':
    # main()
    # show_draw()
    # draw_image_cover([[[66,67,68,69,72,74,78,83,86,89,92,95,97,99,101,103,105,106,109,111,113,115,117,119,121,123,125,127,130,132,134,136,138,142,145,147,149,151,153,155,157,159,160,162,163,165,166,167,168,168,169,170,170,171,172,172,173,174,174,175,176,176,177,178,179,179,180,181,181,182,183,184,186,187,189,190,192,194,196,198,200,202,205,207,209,211,213,216,218,220,222,224,225,227,229,230,232,232,234,235,235,236,237,238,238,239,239,240,241,242,242,243,244,245,245,245,245,246,246,246,247,247,247,247,247,247,247,247,246,245,244,242,241,240,239,238,237,236,235,234,233,232,232,231,230,229,228,227,226,225,224,223,222,221,220,219,217,216,213,211,209,207,205,204,203,201,200,197,195,192,190,189,187,186,185,183,182,180,177,175,171,169,166,163,161,157,153,150,142,139,136,130,128,126,124,121,120,116,114,112,110,108,107,104,101,98,97,96,95,94,93,91,89,88,86,85,83,81,80,79,78,77,75,74,72,72,71,70,70,69,69,69,68,68,67,67,66,66,66,65,65],[94,94,93,92,90,88,86,83,81,79,77,76,75,74,72,72,71,70,69,68,67,66,66,65,65,64,64,63,63,63,62,62,62,62,62,62,62,62,62,62,62,62,63,63,64,66,67,68,70,71,72,73,75,76,77,79,81,82,83,85,87,89,91,92,93,94,95,96,97,98,99,100,101,102,102,103,104,104,105,105,105,106,106,106,106,107,107,107,107,107,107,107,106,105,104,104,103,102,101,100,99,97,96,95,93,92,90,87,86,83,81,80,78,76,75,74,73,72,70,69,68,67,66,65,64,63,62,61,60,60,59,58,57,57,57,57,57,57,57,57,57,57,56,56,56,56,57,58,59,60,61,62,63,64,65,67,70,72,77,80,84,87,90,92,94,96,99,102,105,110,112,113,115,117,118,119,120,121,123,124,125,126,126,126,126,126,126,126,125,124,123,122,121,120,120,119,118,118,117,117,116,116,116,115,114,114,114,113,113,113,112,112,111,110,110,109,108,106,105,104,103,103,102,101,101,100,100,100,99,99,98,97,97,96,96,95,95,94,93,93,92]],[[100],[92]],[[100],[92]],[[91,92,93,94,95,96,97,98,99,100,100,100,99,98,97,96,95,94,94,93,93,93],[94,94,94,94,94,94,94,94,93,93,92,91,90,90,90,90,90,90,91,92,93,94]],[[102,103,103,104,105,105,106,106,107,107,107,107,107,108,108,108,108,108,108,108,108,108,108,108,108,108,108,108,108,107,107,107,106,106,106,105,105,104,104],[75,75,76,78,79,80,81,82,84,85,86,87,88,89,90,91,92,94,95,96,97,98,99,100,101,102,103,104,105,105,106,107,107,108,109,109,110,111,112]]])


def is_point_above_line(point, line):
    x1, y1 = point
    x2, y2, x3, y3 = line
    
    # Check if the line is vertical
    if x3 == x2:
        if max(y2,y3) < y1:
            return True
        else:
            return False
    
    # calculate the slope and intercept of the line
    slope = (y3 - y2) / (x3 - x2)
    intercept = y2 - slope * x2
    
    # calculate the y-coordinate of the line at the point x
    line_y = slope * x1 + intercept
    
    # compare the point y to line_y to determine if the point is above or below the line
    if y1 > line_y:
        return True
    else:
        return False

def is_point_right_line(point, line):
    x1, y1 = point
    x2, y2, x3, y3 = line
    
    # Check if the line is vertical
    if x3 == x2:
        if x1 > x3:
            return True
        else:
            return False
    
    # calculate the slope and intercept of the line
    slope = (y3 - y2) / (x3 - x2)
    intercept = y2 - slope * x2
    
    # calculate the x-coordinate of the line at the point y
    line_x = (y1 - intercept) / slope
    
    # compare the point x to line_x to determine if the point is to the right or left of the line
    if x1 > line_x:
        return True
    else:
        return False

def is_intersect(box, line):
    x1, y1, x2, y2 = box
    x3, y3, x4, y4 = line
    dx = x4 - x3
    dy = y4 - y3

    if dx == 0:
        # line is vertical
        if x1 <= x3 and x3 <= x2:
            # line passes through the box
            if y3 <= y1 and y2 <= y4:
                # box is contained within the line
                return True
            elif y1 <= y3 and y3 <= y2:
                # box intersects the line
                return True
        else:
            return False
    elif dy == 0:
        # line is horizontal
        if y1 <= y3 and y3 <= y2:
            # line passes through the box
            if x3 <= x1 and x2 <= x4:
                # box is contained within the line
                return True
            elif x1 <= x3 and x3 <= x2:
                # box intersects the line
                return True
        else:
            return False
    else:
        # line is not vertical or horizontal
        # check for intersection with box
        # first find the slope and y-intercept of the line
        m = dy / dx
        b = y3 - m * x3

        # next, find the x-coordinate of the intersection with the top of the box
        x = (y1 - b) / m

        # finally, check if the x-coordinate of the intersection is within the bounds of the box
        if x1 <= x and x <= x2:
            return True
        else:
            if max(x1, x2) >= min(x3, x4):
                return False
            else:
                return False

def get_points_from_box(box):
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

def get_line_orientation(line):
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0:
        # line is vertical
        return "vertical"
    else:
        m = dy / dx
        if abs(m) < 1:
            return "horizontal"
        else:
            return "vertical"

def check_box_position(box, line):
    box_points = get_points_from_box(box)
    line_orientation = get_line_orientation(line) 
    default_position = ""
    
    if line_orientation == "horizontal":
        default_position = "below"
    else:
        default_position = "left"

    if is_intersect(box, line):
        return "intersect"

    if line_orientation == "horizontal":
        for point in box_points:
            if is_point_above_line(point, line):
                return "above"
    elif line_orientation == "vertical":
        for point in box_points:
            if is_point_right_line(point, line):
                return "right"
        
    return default_position

def frame_to_timestamp(frame_num, fps):
    total_seconds = frame_num / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    return '{:03d}:{:02d}:{:02d}.{:03d}'.format(hours, minutes, seconds, milliseconds)

if __name__ == '__main__':

    import matplotlib.pyplot as plt # for testing purposes

    def test_check_box_position(box, line):
        x1, y1, x2, y2 = box
        x3, y3, x4, y4 = line
        
        print(get_line_orientation(line))
        
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=None)
        plt.gca().add_patch(rect)
        plt.plot([x3, x4], [y3, y4])
        plt.text((x1 + x2) / 2, (y1 + y2) / 2, check_box_position(box,line), fontsize=12, ha="center", va="center")

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.gca().set_aspect("equal")
        plt.show()

    ## Test case 1: Line is horizontal and intersects with box
    box = [0, 0, 5, 5]
    line = [-10, 2, 10, 2]
    test_check_box_position (box,line)

    # Test case 2: Line is horizontal and box is above the line
    box = [0, 6, 5, 11]
    line = [2, 2, 8, 2]
    test_check_box_position (box,line)

    # Test case 3: Line is horizontal and box is below the line
    box = [-5, -5, 0, 0]
    line = [2, 2, 8, 2]
    test_check_box_position (box,line)

    # Test case 4: Line is vertical and intersects with box
    box = [0, 0, 5, 5]
    line = [2, 2, 2, 8]
    test_check_box_position (box,line)

    # Test case 5: Line is vertical and box is right of the line
    box = [6, 0, 11, 5]
    line = [2, 2, 2, 8]
    test_check_box_position (box,line)

    # Test case 6: Line is vertical and box is left of the line
    box = [-5, -5, 0, 0]
    line = [2, 2, 2, 8]
    test_check_box_position (box,line)

    #Test case 7: Line has slope 1 and intersects with box
    box = [0, 0, 5, 5]
    line = [-10, -10, 10, 10]
    test_check_box_position (box,line)

    #Test case 8: Line has slope -1 and intersects with box
    box = [0, 0, 5, 5]
    line = [10, 10, -10, -10]
    test_check_box_position (box,line)

    #Test case 9: Line has slope 2 and box is to the right of the line
    box = [6, 6, 11, 11]
    line = [0, 0, 10, 20]
    test_check_box_position (box,line)

    #Test case 10: Line has slope -0.5 and box is to the left of the line
    box = [-5, -5, 0, 0]
    line = [0, 0, 10, -5]
    test_check_box_position (box,line)

    #Test case 11: Line has slope 0 and box is to the right of the line
    box = [6, 0, 11, 5]
    line = [0, 2, 0, 8]
    test_check_box_position (box,line)

    #Test case 12: Line has slope 0 and box is to the left of the line
    box = [-5, -5, 0, 0]
    line = [0, 2, 0, 8]
    test_check_box_position (box,line)

    # Test case 1: Frame number = 0, fps = 30
    assert frame_to_timestamp(0, 30) == '000:00:00.000'

    # Test case 2: Frame number = 1, fps = 30
    assert frame_to_timestamp(1, 30) == '000:00:00.333'

    # Test case 3: Frame number = 900, fps = 30
    assert frame_to_timestamp(900, 30) == '000:15:00.000'

    # Test case 4: Frame number = 901, fps = 30
    assert frame_to_timestamp(901, 30) == '000:15:00.333'

    # Test case 5: Frame number = 7200, fps = 24
    assert frame_to_timestamp(7200, 24) == '002:00:00.000'
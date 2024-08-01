import ctypes

# function to get screen size
def get_screen_size():
    user32 = ctypes.windll.user32

    return [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
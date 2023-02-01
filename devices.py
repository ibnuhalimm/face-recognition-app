import cv2

class Devices:
    def list_ports(self):
        non_working_ports = []
        dev_port = 0
        working_ports = []
        available_ports = []

        while len(non_working_ports) < 3:
            camera = cv2.VideoCapture(dev_port)
            if not camera.isOpened():
                non_working_ports.append(dev_port)
                # print("Port %s is not working" %dev_port)
            else:
                is_reading, img = camera.read()
                resolution_width = camera.get(3)
                resolution_height = camera.get(4)
                if is_reading:
                    # print("Port %s is working and reads images (%s x %s)" %(dev_port, resolution_width, resolution_height))
                    working_ports.append(dev_port)
                else:
                    # print("Port %s for camera (%s x %s) is present, but does not reads" %(dev_port, resolution_width, resolution_height))
                    available_ports.append(dev_port)
            dev_port += 1

        return working_ports

def main():
    Devices()

if __name__ == '__main__':
    main()

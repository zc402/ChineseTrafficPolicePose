import numpy as np
import cv2
import os
import glob

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf

THUMBNAIL_PATH = "./temp"
class VideoToTempFile:

    def __init__(self):
        self.num_frame = 0

    def _check_video(self, vcap):
        fps = vcap.get(5)
        if fps != 15:
            raise ValueError("video must be 15 fps")

    def save(self, video):

        if os.path.exists(THUMBNAIL_PATH):
            temp_pngs = glob.glob(os.path.join(THUMBNAIL_PATH, "*.png"))
            [os.remove(p) for p in temp_pngs]
            os.rmdir(THUMBNAIL_PATH)
        os.mkdir(THUMBNAIL_PATH)

        num_list = []  # saved file number
        cap = cv2.VideoCapture(video)
        self._check_video(cap)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            self.num_frame = self.num_frame + 1

            if self.num_frame % 5 == 0:
                thumbnail = cv2.resize(frame, (100, 100))
                savedir = "%d.png" % self.num_frame
                savedir = os.path.join(THUMBNAIL_PATH, savedir)
                cv2.imwrite(savedir, thumbnail)
                num_list.append(self.num_frame)
                print(savedir)

        cap.release()
        return num_list

class LabelUtil:

    def load_label(self, csv_file):
        """
        Label file is a csv file using number to mark gesture for each frame
        example content: 0,0,0,2,2,2,2,2,0,0,0,0,0
        :param csv_file:
        :return: list of int
        """
        with open(csv_file, 'r') as label_file:
            labels = label_file.read()

        labels = labels.split(",")
        labels = [int(l) for l in labels]
        # Labels is a list of int, representing gesture for each frame
        return labels

    def save_label(self, label_list, csv_file):
        """

        :param label_list: a list of int
        :param csv_file:
        :return:
        """
        str_line = ",".join(map(str, label_list))

        with open(csv_file, 'w') as label_file:
            label_file.write(str_line)



class FlowBoxWindow(Gtk.Window):


    def __init__(self, file_numbers):
        """

        :param file_numbers: a ordered list of numbers, referenced to the file name
        """
        Gtk.Window.__init__(self, title="Hello World")

        self.set_border_width(10)
        self.set_default_size(800, 600)

        header = Gtk.HeaderBar(title="Flow Box")
        header.set_subtitle("Sample FlowBox app")
        header.props.show_close_button = True
        self.set_titlebar(header)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        flowbox = Gtk.FlowBox()
        flowbox.set_valign(Gtk.Align.START)
        flowbox.set_max_children_per_line(30)
        flowbox.set_selection_mode(Gtk.SelectionMode.NONE)

        self.create_flowbox(flowbox, file_numbers)

        scrolled.add(flowbox)
        self.add(scrolled)
        self.show_all()

    def thumbnail_onclick(self, button, data):
        frame = data["frame"]
        print(frame)

    # Create a button with picture on it
    def new_thumbnail_button(self, num_frame):

        filepath = os.path.join(THUMBNAIL_PATH, str(num_frame) + ".png")
        button = Gtk.Button()
        img = Gtk.Image.new_from_file(filepath)
        button.add(img)
        button.connect("clicked", self.thumbnail_onclick, {"frame": num_frame})

        return button

    # Fill color to color bar
    def on_draw(self, widget, cr, data):

        context = widget.get_style_context()

        width = widget.get_allocated_width()
        height = widget.get_allocated_height()
        Gtk.render_background(context, cr, 0, 0, width, height)

        r,g,b,a = data['color']
        cr.set_source_rgba(r,g,b,a)
        cr.rectangle(0, 0, width, height)
        cr.fill()

    def create_flowbox(self, flowbox, frame_list):
        """
        Create the flowing box containing picture and color bar
        :param flowbox: parent widget
        :param frame_list: int list contains file number
        :return:
        """

        for num_frame in frame_list:
            grid = Gtk.Grid()
            btn = self.new_thumbnail_button(num_frame)

            area = Gtk.DrawingArea()
            area.set_size_request(20, 20)

            color = Gdk.color_parse('aqua')
            rgba = Gdk.RGBA.from_color(color)
            area.connect("draw", self.on_draw, {'color':rgba})
            grid.add(btn)
            grid.attach_next_to(area, btn, Gtk.PositionType.BOTTOM, 1, 2)

            flowbox.add(grid)


vtf = VideoToTempFile()
numbers = vtf.save("/home/zc/eval6.mp4")

win = FlowBoxWindow(numbers)
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()
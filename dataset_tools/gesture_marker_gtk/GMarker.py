import numpy as np
import cv2
import os
import glob
import argparse
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

        # Remove thumbnail caches
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
                thumbnail = cv2.resize(frame, (150, 150))
                savedir = "%d.png" % self.num_frame
                savedir = os.path.join(THUMBNAIL_PATH, savedir)
                cv2.imwrite(savedir, thumbnail)
                num_list.append(self.num_frame)
                print(savedir)
        cap.release()
        return num_list # numlist Start from 5

    def length(self, video):
        cap = cv2.VideoCapture(video)
        length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return length

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
        str_list = ["%d" % e for e in label_list]
        str_line = ",".join(str_list)

        with open(csv_file, 'w') as label_file:
            label_file.write(str_line)
        print("saved: %s" % csv_file)



class FlowBoxWindow(Gtk.Window):


    def __init__(self, list_label, thumbnail_numbers, csv_path):
        """

        :param list_label: list of label, containing class numbers of each frame.
        :param thumbnail_numbers: a ordered list of numbers, referenced to the file name
        """
        self.list_label = list_label  # A list containing labels.
        self.label_widgets = []  # A list containing widgets, showing labels of each image
        self.thumbnail_numbers = thumbnail_numbers
        self.select1 = None  # First selected picture
        self.select2 = None  # Second selected picture
        self.key_pressed = None  # Pressed key for marking label
        self.flowbox_layout = None  # Needed by updating color of draw area
        self.csv_path = csv_path

    def create_window(self):
        """

        :return:
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

        self.create_flowbox(flowbox, self.thumbnail_numbers)

        scrolled.add(flowbox)
        self.add(scrolled)
        self.connect("key-press-event", self.main_win_key_press)
        self.show_all()

    def main_win_key_press(self):
        # TODO: ctrl-s to save
        pass

    def label_dialog_key_press(self, widget, event):
        key_name = Gdk.keyval_name(event.keyval)
        key_val = event.keyval  # '0': value=48 1:49 9:57
        if 48 <= key_val <= 57:
            label = key_val-48  # label: int number of 0~9
            print("Marked: %d" % label)
            self.key_pressed = label
            widget.destroy()
        else:  # keys other than 0~9 are not supported
            self.key_pressed = None
            print("Invalid key %s" % key_name)

    def create_mark_label_prompt(self):
        self.key_pressed = None  # No key pressed, canceled directly
        dialog = Gtk.MessageDialog(self, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.CANCEL, "Marking: %d to %d" % (self.select1, self.select2))
        dialog.format_secondary_text("Press 0~9 on keyboard to mark a label!")
        dialog.connect("key-press-event", self.label_dialog_key_press)
        dialog.run()  # Block until dialog closed
        dialog.destroy()

    def thumbnail_onclick(self, button, data):
        """
        Clicked on picture
        :param button: widget
        :param data: frame of this picture
        :return:
        """
        frame = data["frame"]

        if self.select1 is None:  # selecting 1st picture
            self.select1 = frame
        elif (self.select1 is not None) and (self.select2 is None):  # selecting 2nd picture
            if frame <= self.select1:  # Selected a previous picture, not allowed
                return
            self.select2 = frame
            self.create_mark_label_prompt()  # key stored in self.key_pressed
            # Two cases: key pressed, or canceled
            if self.key_pressed is not None:
                for i_frame in range(self.select1, self.select2+1, 1):
                    self.list_label[i_frame] = self.key_pressed
            self.select1 = None
            self.select2 = None
            self.key_pressed = None
            LabelUtil().save_label(self.list_label, self.csv_path)

        else:  # Selecting 3rd picture
            pass

        self.flowbox_layout.queue_draw()  # Refresh widgets

    # Create a button with picture on it
    def new_thumbnail_button(self, num_frame):

        filepath = os.path.join(THUMBNAIL_PATH, str(num_frame) + ".png")
        button = Gtk.Button()
        img = Gtk.Image.new_from_file(filepath)
        button.add(img)
        button.connect("clicked", self.thumbnail_onclick, {"frame": num_frame})

        return button

    def update_text_color_by_frame(self, frame, text):
        cls = self.list_label[frame]
        if cls == 0:
            str_color = "white"
            text.set_text(" ")
        else:
            str_color = "aquamarine"
            text.set_text(str(cls))

        color = Gdk.color_parse(str_color)
        rgba = Gdk.RGBA.from_color(color)
        return rgba

    # Draw the color bar, or change it's color
    # Also change the number
    def area_on_draw(self, widget, cr, data):

        context = widget.get_style_context()
        width = widget.get_allocated_width()
        height = widget.get_allocated_height()
        Gtk.render_background(context, cr, 0, 0, width, height)

        frame = data['frame']
        widget_label = data['widget_label']
        r,g,b,a = self.update_text_color_by_frame(frame, widget_label)
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

            widget_cls_label = Gtk.Label()
            widget_cls_label.set_text("2")
            widget_cls_label.set_size_request(20, 20)
            widget_cls_label.connect("draw", self.area_on_draw, {'frame': num_frame, 'widget_label': widget_cls_label})
            # Add drawing area
            grid.add(btn)
            grid.attach_next_to(widget_cls_label, btn, Gtk.PositionType.BOTTOM, 1, 2)

            flowbox.add(grid)
            self.flowbox_layout = flowbox

    def release(self):
        del self.flowbox_layout

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="video file to be annotated")
    parser.add_argument("csv", type=str, help="csv file")
    args = parser.parse_args()
    if not os.path.exists(args.video):
        raise FileNotFoundError()
    vtf = VideoToTempFile()
    numbers = vtf.save(args.video)

    if os.path.exists(args.csv):
        # Read csv
        label_list = LabelUtil().load_label(args.csv)
    else: # File not exist
        length = vtf.length(args.video)
        label_list = np.zeros(int(length)).tolist()

    win = FlowBoxWindow(label_list, numbers, args.csv)
    win.create_window()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()

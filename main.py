"""
This program is used to calculated propagation of optic field and display it.

At current moment, program only propagate optic field between -30 to 30 micrometers from loaded optic field, with step
equal to 5.

For this program to work, you should have installed filebrowser widget for kivy to do so:
garden install filebrowser --kivy		<-- this will provide globally and put this into your Lib\site-packages dir
garden install filebrowser --app		<-- this will install only locally, for this app
"""

import matplotlib
import threading
import numpy as np
import math
from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.uix.widget import Widget
from kivy.uix.slider import Slider
from kivy.uix.popup import Popup
from kivy.uix.settings import SettingsWithSidebar
from kivy.graphics.texture import Texture
from functools import partial
from scipy import fftpack, io
from skimage.restoration import unwrap_phase
from scipy.misc import imread
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.use('Agg')
# matplotlib.use should be called before importing any backend or pyplot.
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

__author__ = "Damian Korbuszewski"
__version__ = "0.1"

z_vec = list(range(-30, 31, 5))

# todo settings for z_vec and steps??

json = '''
[
  {
    "desc": "Enter lambda of used light in micrometers",
    "key": "lambda",
    "section": "Optics",
    "title": "lambda",
    "type": "numeric"
  },
  {
    "desc": "Enter the refractive index of medium",
    "key": "n0",
    "section": "Optics",
    "title": "n0",
    "type": "numeric"
  },
  {
    "desc": "Enter the size of camera pixel in micrometers",
    "key": "pixel_size",
    "section": "Optics",
    "title": "Pixel size",
    "type": "numeric"
  },
  {
    "desc": "enter magnification of optical system",
    "key": "magnification",
    "section": "Optics",
    "title": "Magnification",
    "type": "numeric"
  }
]
'''


# could make just .json file to create settings.


def propagate2d(ui, z, lam, n0, dx):
    """
    Function used to propagate optic field to other location, before or after current position
    :param ui: 2D complex array with registered optic field
    :param z: float/int value, distance how far we want to propagate field, if negative, function will return field
    before current position
    :param lam: float with length of light in um
    :param n0: float with refractive index of medium
    :param dx: float with   pixel size divided by magnification of optic system
    :return: uo, optic field at desire distance from current location
    """
    k = 2 * math.pi / lam
    Ny, Nx = ui.shape
    dfx = 1 / Nx / dx
    dfy = 1 / Ny / dx
    fx2 = np.linspace(-int(Nx / 2) * dfx, int(Nx / 2) * dfx, Nx)
    fx2 = np.power(fx2, 2)
    fy2 = np.linspace(-int(Ny / 2) * dfy, int(Ny / 2) * dfy, Ny)
    fy2 = np.power(fy2, 2)

    kernel = np.dot(np.ones([Ny, 1]), fx2[np.newaxis]) + np.dot(fy2[np.newaxis].T, np.ones([1, Nx]))
    kernel = np.power(n0, 2) - np.power(lam, 2) * kernel
    mask = np.ones([Ny, Nx])
    mask[kernel < 0] = 0
    kernel[kernel < 0] = 0

    if z < 0:
        kernel = np.exp(1j * k * z * np.sqrt(np.array(kernel, dtype=complex)))
        ftu = fftpack.fftshift(fftpack.fft2(np.conj(ui)))
        ftu = ftu * kernel * mask
        uo = np.conj(fftpack.ifft2(fftpack.ifftshift(ftu)))
    else:
        kernel = np.exp(-1j * k * z * np.sqrt(np.array(kernel, dtype=complex)))
        ftu = fftpack.fftshift(fftpack.fft2(ui))
        ftu = ftu * kernel * mask
        uo = fftpack.ifft2(fftpack.ifftshift(ftu))
    return uo


class LoadDialog(Popup):
    """Simple class used to make popup with FileBrowser widget"""
    load = ObjectProperty(None)


class PopupBox(Popup):
    """Class to display popup when program is calculating optic field in other positions, and unwrapping phase"""
    pass


class SpecSlider(Slider):
    """Simple class to override default on_touch_up method of slider"""

    def on_touch_up(self, touch):
        """
        Override on_touch_up method of Slider class, otherwise is hard to make changes, only when we stop using slider,
        and with default events/binds its only possible with each change of value.
        It handles event, and after that will call method to update image widget
        """
        if touch.grab_current is not self:
            return
        touch.ungrab(self)  # Without it, app will not respond to new touches.
        app = App.get_running_app()
        root = app.root
        ampli = root.ids.amplitude_button.state  # Status of toggle button.
        root.update_image(self.value, ampli)  # Call method to update image.


class SelectMatVariable(Popup):
    """Class to display popup when loading .mat file with many variables."""
    values = ObjectProperty(None)


class WrongFileDialog(Popup):
    """Class to display popup when user try to open unsupported file. Appearance is defined in kv file."""

    def reload(self):
        """It's called when user press OK button, on popup. It's closed this popup, and reload LoadDialog popup."""
        self.dismiss()
        app = App.get_running_app()  # Get instance of current app.
        root = app.root
        root.popup.open()


class MainWidget(Widget):
    def __init__(self):
        super(MainWidget, self).__init__()
        self.up1 = []
        self.phase = []
        self.popup = LoadDialog(load=self.load)
        self.popup2 = WrongFileDialog()
        self.popup3 = SelectMatVariable(values='something')  # Not nice at all.
        self.popup4 = PopupBox()
        self.pb = self.popup4.ids.loading_progress
        self.mat_structure = []
        self.variable_tuple = []
        self.file2open = ' '
        self.dx = 0

    def mat_callback(self, instance):
        """
        called when user closed the popup in which he selected variable from matlab file. Load matlab file, load
        selected variable and call prepare_data
        """
        ui = io.loadmat(self.file2open)
        ui = ui[self.popup3.ids.mat_spinner.text]
        self.popup.dismiss()
        self.popup4.open()
        mythread = threading.Thread(target=partial(self.propagate_all, ui))
        mythread.start()

    def load(self, file2open):
        """
        Function to check if selected file is supported, and open it or call method to choose variable in multiple
        variables files.
        :param file2open: string with path for app to open
        """

        # stupid approach based on file extension.
        if file2open[-4:].lower() == '.mat':
            self.mat_structure = io.whosmat(file2open)
            if len(self.mat_structure) == 1:
                ui = io.loadmat(file2open)
                ui = ui[self.mat_structure[0][0]]
                self.popup.dismiss()
                self.popup4.open()
                mythread = threading.Thread(target=partial(self.propagate_all, ui))
                mythread.start()
                # open .mat file with single variable.
            if len(self.mat_structure) > 1:
                self.variable_tuple = tuple(x[0] for x in self.mat_structure)
                self.popup3 = SelectMatVariable(values=self.variable_tuple)
                self.popup3.open()
                self.file2open = file2open
                self.popup3.bind(on_dismiss=self.mat_callback)
                # Display popup to select variable from multi variable .mat file.
        elif file2open[-4:].lower() == '.jpg' or file2open[-4:].lower() == '.bmp' or file2open[-4:].lower() == '.png':
            ui = imread(file2open, flatten=True).astype(np.float32)
            self.popup.dismiss()
            self.popup4.open()
            mythread = threading.Thread(target=partial(self.propagate_all, ui))
            mythread.start()
            # Open image and treat it as optic field. don't ask why.
        else:
            self.popup.dismiss()
            self.popup2.open()
            # Close file browser widget and open popup with information that selected file is not supported.

    def propagate_all(self, ui):
        """
        It propagates optic field in some range (-30, 30) with 5 as step.
        :param ui: numpy 2d array with optic field.
        """
        app = App.get_running_app()
        # Now we will get set values in settings by user.
        lam = float(app.config.getdefault('Optics', 'lambda', 0))
        n0 = float(app.config.getdefault('Optics', 'n0', 0))
        pixel = float(app.config.getdefault('Optics', 'pixel_size', 0))
        magn = float(app.config.getdefault('Optics', 'magnification', 0))
        self.dx = pixel / magn

        # Create empty numpy array as 3d array, in which are stored propagated fields.
        size = list(ui.shape)
        size.append(1)
        self.up1 = np.empty(size)

        # In this for loop we stack in empty array values of propagated field and increase value of progress bar.
        self.popup4.title = 'Calculating'  # Make sure that after first loading we still have proper title.
        self.pb.value = 0  # And proper starting value.
        for i in range(-30, 31, 5):
            self.up1 = np.dstack((self.up1, propagate2d(ui, i, lam, n0, self.dx)))
            self.pb.value += 1
        self.up1 = np.delete(self.up1, 0, axis=2)  # Delete empty slice array from array with fields.

        # Now in array each 2d slice is corresponded to optic field in one place.
        self.popup4.title = 'Unwraping'  # Change name of popup dialog.
        self.pb.value = 0  # Reset value of progress bar.
        self.phase = np.angle(self.up1)
        # Unwrap phase for each slice in 3d array.
        for i in range(13):
            self.phase[:, :, i] = unwrap_phase(self.phase[:, :, i])
            self.pb.value += 1

        self.ids.position_slider.disabled = False  # Unlock slider after reading first file.
        self.ids.phase_button.disabled = False
        self.ids.amplitude_button.disabled = False  # Unlock toggle buttons.
        self.up1 = np.abs(self.up1)  # Now we have two 3d arrays, one with amplitude and other with phase.
        self.update_image(self.ids.position_slider.value, self.ids.amplitude_button.state)
        # Above line don't update image, it displays it as black, and I don't know why.
        self.popup4.dismiss()

    def update_image(self, value, state):
        """
        Prepare imshow plot. After creating it, call make_texture method to update texture of imshow_image
        :param value: value of position_slider,
        :param state: state of one of toggle buttons on app, based on it status it's plotting amplitude or phase.
        """
        i = z_vec.index(value)
        plt.close('all')  # Without it, we will  just eat memory like crazy.
        fig = plt.gcf()
        fig.set_dpi(400)  # Plot will have 3200x2400 resolution.

        if state == 'down':
            plt.imshow(self.up1[:, :, i], cmap='gray')
        else:
            plt.imshow(self.phase[:, :, i], cmap='gray')

        # Make colorbar similar in size as height of imshow plot.
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)

        # Scale x and y axis based on pixel size of camera and magnification of optic system.
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x * self.dx).rstrip('0').rstrip('.'))
        ax.xaxis.set_major_formatter(ticks_x)
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x * self.dx).rstrip('0').rstrip('.'))
        ax.yaxis.set_major_formatter(ticks_y)

        # Move x axis on top.
        ax.set_xlabel('micrometers')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

        fig.tight_layout()  # Less margin around figure and between imshow and colorbar.
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        s = canvas.tostring_rgb()
        x, y = fig.get_size_inches() * fig.dpi
        self.make_texture(s, x, y)

    def make_texture(self, s, x, y):
        """
        Function to update texture of imshow_image
        :param s: string with colors of created canvas
        :param x: width of canvas
        :param y: height of canvas.
        """
        tex = Texture.create(size=(x, y), colorfmt='rgb')  # Create texture of proper size, and color format.
        tex.blit_buffer(s, bufferfmt="ubyte", colorfmt="rgb")  # Transfer string with color to texture.
        tex.flip_vertical()  # Data is upside down after blit_buffer.
        self.ids.imshow_image.texture = tex


class PropagateApp(App):
    """Main class of program, used to make settings panel"""
    use_kivy_settings = False  # we don't want to see default kivy settings, don't need it

    def build(self):
        propagate = MainWidget()
        self.settings_cls = SettingsWithSidebar  # set settings class, used to change appearance of setting screen
        return propagate

    def build_config(self, config):
        """Set default option for settings"""
        config.setdefaults('Optics', {'lambda': .6328, 'n0': 1.333, 'pixel_size': 3.45, 'magnification': 30.5})

    def build_settings(self, settings):
        """Add settings panel to settings screen, based on json"""
        settings.add_json_panel('Optics', self.config, data=json)



if __name__ == '__main__':
    PropagateApp().run()

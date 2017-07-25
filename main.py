import matplotlib
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.uix.slider import Slider
from kivy.uix.popup import Popup
from kivy.uix.settings import SettingsWithSidebar
from kivy.app import App
from kivy.graphics.texture import Texture
from kivy.factory import Factory
from kivy.properties import ObjectProperty
import threading
from functools import partial
from scipy import fftpack, io
import numpy as np
import math
from skimage.restoration import unwrap_phase
from scipy.misc import imread

matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

z_vec = list(range(-30, 31, 5))

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

# todo settings for z_vec and steps??
# todo check if correct file is selected, if not display popup



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


class LoadDialog(FloatLayout):
    """Simple class used to make popup with FileBrowser widget"""
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class PopupBox(Popup):
    """Class to display popup when program is calculating optic field in other positions, and unwrapping phase"""
    pass


class SpecSlider(Slider):
    """Simple class to override default on_touch_up method of slider"""

    def on_touch_up(self, touch):
        """
        Override on_touch_up method of Slider class, otherwise is hard to make changes, only when we stop using slider,
        and with default events/binds it's only possible with each change of value.
        It handle event, and after that will call method to update image widget
        """
        if touch.grab_current is not self:
            return
        touch.ungrab(self)  # without it, app will not respond to new touches
        app = App.get_running_app()
        root = app.root
        ampli = root.ids.amplitude_button.state  # status of toggle button
        root.update_image(self.value, ampli)  # call method to update image


class SelectMatVariable(Popup):
    """
    Class to display popup when loading .mat file with many variables
    """
    values = ObjectProperty(None)


class WrongFileDialog(Popup):
    """
    Class to display popup when user try to open unsupported file. Appearance is defined in kv file
    """

    def reload(self):
        """
        It's called when user press ok button, on popup. It's closed this popup, and reload LoadDialog popup.
        """
        self.dismiss()
        app = App.get_running_app()  # get instance of current app
        root = app.root
        root._popup.open()


class MainWidget(Widget):
    def __init__(self, **kwargs):
        super(MainWidget, self).__init__()
        self.up1 = []

    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)

    def dismiss_popup(self):
        """Method to close popup with filebrowser widget"""
        self._popup.dismiss()

    def show_load(self):
        """
        Create and open popup with filebrowser widget, and bind events of filebrowser widget with methods from this
        class.
        """
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(1, 1))
        self._popup.open()

    def show_loading(self):
        """
        Display popup when program is propagatic optic field, and unwrapping phase, user should now that program\
        is working
        """
        self._content = PopupBox()
        self._content.open()
        self.pb = self._content.ids.loading_progress

    def mat_callback(self, instance):
        """
        called when user closed the popup in which he selected variable from matlab file. Load matlab file, load selected
        variable and call prepare_data
        """
        ui = io.loadmat(self.file2open)
        ui = ui[self._popup3.ids.mat_spinner.text]
        self.dismiss_popup()
        self.show_loading()
        mythread = threading.Thread(target=partial(self.propagate_all, ui))
        mythread.start()

    def load(self, file2open):
        if file2open[-4:].lower() == '.mat':
            self.mat_structure = io.whosmat(file2open)
            if len(self.mat_structure) == 1:
                ui = io.loadmat(file2open)
                ui = ui[self.mat_structure[0][0]]
                self.dismiss_popup()
                self.show_loading()
                mythread = threading.Thread(target=partial(self.propagate_all, ui))
                mythread.start()
            if len(self.mat_structure) > 1:
                self.variable_tuple = tuple(x[0] for x in self.mat_structure)
                self._popup3 = SelectMatVariable(values=self.variable_tuple)
                self._popup3.open()
                self.file2open = file2open
                self._popup3.bind(on_dismiss=self.mat_callback)
        elif file2open[-4:].lower() == '.jpg' or file2open[-4:].lower() == '.bmp' or file2open[-4:].lower() == '.png':
            ui = imread(file2open, flatten=1).astype(np.float32)
            self.dismiss_popup()
            self.show_loading()
            mythread = threading.Thread(target=partial(self.propagate_all, ui))
            mythread.start()
        else:
            self.dismiss_popup()
            self._popup2 = WrongFileDialog()
            self._popup2.open()

    def propagate_all(self, ui):
        app = App.get_running_app()
        # now we will get set values in settings by user
        lam = float(app.config.getdefault('Optics', 'lambda', 0))
        n0 = float(app.config.getdefault('Optics', 'n0', 0))
        pixel = float(app.config.getdefault('Optics', 'pixel_size', 0))
        magn = float(app.config.getdefault('Optics', 'magnification', 0))
        self.dx = pixel / magn
        # create empty numpy array as 3d array, in which are stored propagated fields
        size = list(ui.shape)
        size.append(1)
        self.up1 = np.empty(size)
        # in this for loop we stack in empty array values of propagated field and increase value of progress bar
        for i in range(-30, 31, 5):
            self.up1 = np.dstack((self.up1, propagate2d(ui, i, lam, n0, self.dx)))
            self.pb.value += 1
        self.up1 = np.delete(self.up1, 0, axis=2)  # delete empty slice array from array with fields
        # now in array each 2d slice is corresponded to optic field in one place
        self._content.title = 'Unwraping'  # change name of popup dialog
        self.pb.value = 0  # reset value of progress bar
        # unwrap phase for each slice in 3d array
        self.phase = np.angle(self.up1)
        for i in range(13):
            self.phase[:, :, i] = unwrap_phase(self.phase[:, :, i])
            self.pb.value += 1
        self.ids.position_slider.disabled = False
        self.ids.phase_button.disabled = False  # unlock slider if this is first read file
        self.up1 = np.abs(self.up1)  # now we have two 3d arrays, one with amplitude and other with phase
        self.ids.amplitude_button.disabled = False  # unlock toogle buttons
        self.update_image(self.ids.position_slider.value, self.ids.amplitude_button.state)
        self._content.dismiss()

    def update_image(self, value, state):
        i = z_vec.index(value)
        plt.close('all')  # without it we will use just eat memory like crazy
        fig = plt.gcf()
        fig.set_dpi(400)

        if state == 'down':
            plt.imshow(self.up1[:, :, i], cmap='gray')
        else:
            plt.imshow(self.phase[:, :, i], cmap='gray')
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x * self.dx).rstrip('0').rstrip('.'))
        ax.xaxis.set_major_formatter(ticks_x)
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x * self.dx).rstrip('0').rstrip('.'))
        ax.yaxis.set_major_formatter(ticks_y)
        ax.set_xlabel('micrometers')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        fig.tight_layout()
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        s = canvas.tostring_rgb()
        x, y = fig.get_size_inches() * fig.dpi
        self.make_texture(s, x, y)

    def make_texture(self, s, x, y):
        tex = Texture.create(size=(x, y), colorfmt='rgb')
        tex.blit_buffer(s, bufferfmt="ubyte", colorfmt="rgb")
        tex.flip_vertical()
        self.ids.imshow_image.texture = tex


class PropagateApp(App):
    use_kivy_settings = False

    def build(self):
        propagate = MainWidget()
        self.settings_cls = SettingsWithSidebar
        return propagate

    def build_config(self, config):
        config.setdefaults('Optics', {'lambda': .6328, 'n0': 1.333, 'pixel_size': 3.45, 'magnification': 30.5})

    def build_settings(self, settings):
        settings.add_json_panel('Optics', self.config, data=json)


Factory.register('PropagateApp', cls=PropagateApp)
Factory.register('LoadDialog', cls=LoadDialog)

if __name__ == '__main__':
    PropagateApp().run()

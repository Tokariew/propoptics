from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.widget import Widget
from kivy.uix.slider import Slider
from kivy.uix.popup import Popup
from kivy.app import App
import threading
from functools import partial
from scipy import fftpack, io
import numpy as np
import os
import math
from image_widget import ImDisplay
from matplotlib import cm

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

# todo settings or something else to give lambde, n0 i dx, może zakres propagacji, kroki?
# todo check if correct file is selected, if not display popup
# todo change filechooser to filebrowser from garden otherwise we can't access other devices
cmap = cm.gray(np.arange(256))
cmap = cmap[:, 0:3]
cmap = cmap.ravel()
cmap = (255 * cmap).astype(np.int32)
img = np.random.rand(1024, 1024)
lam = .6328
n0 = 1.333
dx = 3.45 / 30.5
z_vec = list(range(-30, 31, 5))


def propagate2d(ui, z, lam, n0, dx):
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
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class PopupBox(Popup):
    pass


class SpecSlider(Slider):
    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return
        touch.ungrab(self)
        app = App.get_running_app()
        root = app.root
        ampli = root.ids.amplitude_button.state
        root.update_image(self.value, ampli)


class MainWidget(Widget):
    def __init__(self, **kwargs):
        super(MainWidget, self).__init__()
        self.boxloy = self.ids.mainbox.__self__
        self.wrong = ImDisplay(size_hint=(1., .1,), height=600, width=600)
        self.wrong.create_im(img, cmap)
        self.boxloy.add_widget(self.wrong)
        self.up1 = []

    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_loading(self):
        self._content = PopupBox()
        self._content.open()
        self.pb = self._content.ids.loading_progress

    def load(self, path, filename):
        file2open = os.path.join(path, filename[0])
        tmp = io.loadmat(file2open)
        ui = tmp['u']
        self.dismiss_popup()
        self.show_loading()
        mythread = threading.Thread(target=partial(self.propagate_all, ui))
        mythread.start()

    def propagate_all(self, ui):
        app = App.get_running_app()
        lam = float(app.config.getdefault('Optics', 'lambda', 0))
        n0 = float(app.config.getdefault('Optics', 'n0', 0))
        pixel = float(app.config.getdefault('Optics', 'pixel_size', 0))
        magn = float(app.config.getdefault('Optics', 'magnification', 0))
        dx = pixel / magn
        size = list(ui.shape)
        size.append(1)
        self.up1 = np.empty(size)
        for i in range(-30, 31, 5):
            self.up1 = np.dstack((self.up1, propagate2d(ui, i, lam, n0, dx)))
            print('calculated: {}'.format(i))
            self.pb.value += 1
        self.up1 = np.delete(self.up1, 0, axis=2)
        self.ids.position_slider.disabled = False
        self.ids.phase_button.disabled = False
        self._content.dismiss()
        self.update_image(self.ids.position_slider.value, self.ids.amplitude_button.state)
        self.ids.amplitude_button.disabled = False

    def update_image(self, value, state):
        i = z_vec.index(value)
        if state == 'down':
            self.wrong.create_im(np.abs(self.up1[:, :, i]), cmap)
        else:
            self.wrong.create_im(np.angle(self.up1[:, :, i]), cmap)


class PropagateApp(App):
    use_kivy_settings = False

    def build(self):
        propagate = MainWidget()
        return propagate

    def build_config(self, config):
        config.setdefaults('Optics', {'lambda': .6328, 'n0': 1.333, 'pixel_size': 3.45, 'magnification': 30.5})

    def build_settings(self, settings):
        settings.add_json_panel('Optics', self.config, data=json)


Factory.register('PropagateApp', cls=PropagateApp)
Factory.register('LoadDialog', cls=LoadDialog)

if __name__ == '__main__':
    PropagateApp().run()


#https://www.youtube.com/watch?v=VBokjWj_cEA

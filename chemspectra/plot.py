from bokeh.palettes import Category10_10
from bokeh.plotting import figure


def plot_spectra(data):
    wavelengths = data.columns.tolist()
    wavelengths = list(map(float, wavelengths))
    samples = data.index.tolist()

    p = figure(title="Mediterranean Honeys FTIR", x_axis_label='Wavelength', y_axis_label='Intensity',
               width=1600, height=450)
    colors = Category10_10

    for i, sample in enumerate(samples):
        intensity_values = data.loc[sample].tolist()
        intensity_values = list(map(float, intensity_values))
        p.line(x=wavelengths, y=intensity_values, line_width=2, color=colors[i % len(colors)])

    return p
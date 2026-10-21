import io
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.pilot_selection.analysis import complete_weeks, prepare_panel
from src.pilot_selection.config import COUNT, PILOTS
from src.pilot_selection.plots import figure_bytes, weekly_figure


def test_plot_uses_original_rows_order_and_independent_axes(synthetic_master):
    complete = complete_weeks(prepare_panel(synthetic_master))
    fig, checks = weekly_figure(complete)
    try:
        assert len(fig.axes) == 6
        assert fig.get_size_inches().tolist() == [12.8, 8.1]
        for i, (ax, pilot) in enumerate(zip(fig.axes, PILOTS)):
            group = complete.loc[complete.municipality_code.eq(pilot.code)].sort_values("week_start")
            assert len(ax.lines) == 1
            assert pilot.code in ax.get_title(loc="left")
            assert ax.lines[0].get_color() == pilot.color
            assert ax.get_yscale() == "linear"
            assert ax.get_ylim()[0] == 0
            assert len(ax.lines[0].get_ydata()) == 991
            np.testing.assert_array_equal(ax.lines[0].get_ydata(), group[COUNT].to_numpy())
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), group.week_start.to_numpy())
            for prior in fig.axes[:i]:
                assert not ax.get_shared_y_axes().joined(ax, prior)
        assert all(row["source_total"] == row["plotted_total"] for row in checks)
    finally:
        plt.close(fig)


def test_png_and_svg_exports(synthetic_master):
    files, checks = figure_bytes(complete_weeks(prepare_panel(synthetic_master)))
    assert len(checks) == 6
    with Image.open(io.BytesIO(files["weekly_comparison_6.png"])) as image:
        assert image.size == (3840, 2430)
        assert abs(image.info["dpi"][0] - 300) < .01
        image.verify()
    root = ET.fromstring(files["weekly_comparison_6.svg"])
    assert not root.findall('.//{http://www.w3.org/2000/svg}image')
    ids = {node.attrib.get("id") for node in root.iter()}
    assert all("serie_" + pilot.code in ids for pilot in PILOTS)

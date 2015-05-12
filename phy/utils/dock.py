# -*- coding: utf-8 -*-

"""Qt dock window."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import sys
import contextlib
from collections import defaultdict

from vispy import app

from ._misc import _is_interactive
from .logging import info, warn


# -----------------------------------------------------------------------------
# PyQt import
# -----------------------------------------------------------------------------

_PYQT = False
try:
    from PyQt4 import QtCore, QtGui
    from PyQt4.QtGui import QMainWindow
    _PYQT = True
except ImportError:
    try:
        from PyQt5 import QtCore, QtGui
        from PyQt5.QtGui import QMainWindow
        _PYQT = True
    except ImportError:
        pass


def _check_qt():
    if not _PYQT:
        warn("PyQt is not available.")
        return False
    return True


if not _check_qt():
    QMainWindow = object  # noqa


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _title(widget):
    return str(widget.windowTitle()).lower()


def _create_web_view(html=None):
    from PyQt4.QtWebKit import QWebView
    view = QWebView()
    if html:
        view.setHtml(html)
    return view


# -----------------------------------------------------------------------------
# Dock main window
# -----------------------------------------------------------------------------

class DockWindow(QMainWindow):
    """A Qt main window holding docking Qt or VisPy widgets."""
    def __init__(self,
                 position=None,
                 size=None,
                 title=None,
                 ):
        super(DockWindow, self).__init__()
        if title is None:
            title = 'phy'
        self.setWindowTitle(title)
        if position is not None:
            self.move(position[0], position[1])
        if size is not None:
            self.resize(QtCore.QSize(size[0], size[1]))
        self.setObjectName(title)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.setDockOptions(QtGui.QMainWindow.AllowTabbedDocks |
                            QtGui.QMainWindow.AllowNestedDocks |
                            QtGui.QMainWindow.AnimatedDocks
                            )
        self._on_show = None
        self._on_close = None

    # Events
    # -------------------------------------------------------------------------

    def on_close(self, func):
        """Register a callback function when the window is closed."""
        self._on_close = func

    def on_show(self, func):
        """Register a callback function when the window is shown."""
        self._on_show = func

    def closeEvent(self, e):
        """Qt slot when the window is closed."""
        if self._on_close:
            self._on_close()
        super(DockWindow, self).closeEvent(e)

    def show(self):
        """Show the window."""
        if self._on_show:
            self._on_show()
        super(DockWindow, self).show()

    # Actions
    # -------------------------------------------------------------------------

    def add_action(self,
                   name,
                   callback=None,
                   shortcut=None,
                   checkable=False,
                   checked=False,
                   ):
        """Add an action with a keyboard shortcut."""
        action = QtGui.QAction(name, self)
        action.triggered.connect(callback)
        action.setCheckable(checkable)
        action.setChecked(checked)
        if shortcut:
            if not isinstance(shortcut, (tuple, list)):
                shortcut = [shortcut]
            for key in shortcut:
                action.setShortcut(key)
        self.addAction(action)
        return action

    def shortcut(self, name, key=None):
        """Decorator to add a global keyboard shortcut."""
        def wrap(func):
            self.add_action(name, shortcut=key, callback=func)
            setattr(self, name, func)
        return wrap

    # Views
    # -------------------------------------------------------------------------

    def add_view(self,
                 view,
                 title='view',
                 position='right',
                 closable=True,
                 floatable=True,
                 floating=None,
                 **kwargs):
        """Add a widget to the main window."""

        if isinstance(view, app.Canvas):
            view = view.native

        # Create the dock widget.
        dockwidget = QtGui.QDockWidget(self)
        dockwidget.setObjectName(title)
        dockwidget.setWindowTitle(title)
        dockwidget.setWidget(view)

        # Set dock widget options.
        options = QtGui.QDockWidget.DockWidgetMovable
        if closable:
            options = options | QtGui.QDockWidget.DockWidgetClosable
        if floatable:
            options = options | QtGui.QDockWidget.DockWidgetFloatable

        dockwidget.setFeatures(options)
        dockwidget.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea |
                                   QtCore.Qt.RightDockWidgetArea |
                                   QtCore.Qt.TopDockWidgetArea |
                                   QtCore.Qt.BottomDockWidgetArea
                                   )

        q_position = {
            'left': QtCore.Qt.LeftDockWidgetArea,
            'right': QtCore.Qt.RightDockWidgetArea,
            'top': QtCore.Qt.TopDockWidgetArea,
            'bottom': QtCore.Qt.BottomDockWidgetArea,
        }[position]
        self.addDockWidget(q_position, dockwidget)
        if floating is not None:
            dockwidget.setFloating(floating)
        dockwidget.show()
        return dockwidget

    def list_views(self, title=''):
        """List all views which title start with a given string."""
        children = self.findChildren(QtGui.QWidget)
        return [child for child in children
                if isinstance(child, QtGui.QDockWidget) and
                _title(child).startswith(title) and
                child.isVisible() and
                child.width() >= 10 and
                child.height() >= 10]

    def view_counts(self):
        """Return the number of opened views."""
        views = self.list_views()
        counts = defaultdict(lambda: 0)
        for view in views:
            counts[_title(view)] += 1
        return dict(counts)

    # State
    # -------------------------------------------------------------------------

    def save_geometry_state(self):
        """Return picklable geometry and state of the window and docks.

        This function can be called in `on_close()`.

        """
        return {
            'geometry': self.saveGeometry(),
            'state': self.saveState(),
            'view_counts': self.view_counts(),
        }

    def restore_geometry_state(self, gs):
        """Restore the position of the main window and the docks.

        The dock widgets need to be recreated first.

        This function can be called in `on_show()`.

        """
        self.restoreGeometry((gs['geometry']))
        self.restoreState((gs['state']))


# -----------------------------------------------------------------------------
# Qt app and event loop integration with IPython
# -----------------------------------------------------------------------------

_APP = None
_APP_RUNNING = False


def _close_qt_after(window, duration):
    """Close a Qt window after a given duration."""
    def callback():
        window.close()
    QtCore.QTimer.singleShot(int(1000 * duration), callback)


def _try_enable_ipython_qt():
    """Try to enable IPython Qt event loop integration.

    Returns True in the following cases:

    * python -i test.py
    * ipython -i test.py
    * ipython and %run test.py

    Returns False in the following cases:

    * python test.py
    * ipython test.py

    """
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except ImportError:
        return False
    if not _is_interactive():
        return False
    if ip:
        ip.enable_gui('qt')
        global _APP_RUNNING
        _APP_RUNNING = True
        return True
    return False


def enable_qt():
    if not _check_qt():
        return
    try:
        from IPython import get_ipython
        ip = get_ipython()
        ip.enable_gui('qt')
        global _APP_RUNNING
        _APP_RUNNING = True
        info("Qt event loop activated.")
    except:
        warn("Qt event loop not activated.")


def start_qt_app():
    """Start a Qt application if necessary.

    If a new Qt application is created, this function returns it.
    If no new application is created, the function returns None.

    """
    # Only start a Qt application if there is no
    # IPython event loop integration.
    if not _check_qt():
        return
    global _APP
    if _try_enable_ipython_qt():
        return
    app.use_app("pyqt4")
    if QtGui.QApplication.instance():
        _APP = QtGui.QApplication.instance()
        return
    if _APP:
        return
    _APP = QtGui.QApplication(sys.argv)
    return _APP


def run_qt_app():
    """Start the Qt application's event loop."""
    global _APP_RUNNING
    if not _check_qt():
        return
    if _APP is not None and not _APP_RUNNING:
        _APP_RUNNING = True
        _APP.exec_()
    if not _is_interactive():
        _APP_RUNNING = False


@contextlib.contextmanager
def qt_app():
    """Context manager to ensure that a Qt app is running."""
    if not _check_qt():
        return
    start_qt_app()
    yield
    run_qt_app()
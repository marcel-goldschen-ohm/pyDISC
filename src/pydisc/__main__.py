from pydisc.DISC import DISCO
from qtpy.QtWidgets import QApplication, QMessageBox


def main():
    app = QApplication()
    ui = DISCO()
    ui.setWindowTitle('DISC')
    ui.show()

    # # MacOS Magnet warning
    # import platform
    # if platform.system() == 'Darwin':
    #     QMessageBox.warning(ui, 'Magnet Warning', 'If you are using the window management software Magnet, please disable it for this app to work properly.')

    # # load example data?
    # answer = QMessageBox.question(ui, 'Example?', 'Load example data?')
    # if answer == QMessageBox.StandardButton.Yes:
    #     load_example(ui)
    
    app.exec()


# def load_example(ui: DISCO):
#     pass


if __name__ == '__main__':
    main()

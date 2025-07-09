"""
Global configuration
"""
# variables to control all map fonts globally
_fontfamily = "Times New Roman"
_mathtext_fontset = "stix"
_mathtext_tt = "Times New Roman"


def set_font(font):
    """
    Function to control the font of all images globally.
    Parameters:
        font (str): the font of the image. 
                    Supported options include 'Times New Roman', 'Helvetica', 'Arial',
                    'Georgia', 'Garamond', 'Verdana', 'Calibri', 'Roboto', 'Courier New',
                    'Consolas'.
    """
    global _fontfamily, _mathtext_fontset, _mathtext_tt
    
    font_case_insensitive = font.lower()
    if font_case_insensitive == "times new roman":
        _fontfamily = "Times New Roman"
        _mathtext_fontset = "stix"
        _mathtext_tt = "Times New Roman"
    elif font_case_insensitive == "helvetica":
        _fontfamily = "Helvetica"
        _mathtext_fontset = "stixsans"
        _mathtext_tt = "Helvetica"
    elif font_case_insensitive == "arial":
        _fontfamily = "Arial"
        _mathtext_fontset = "custom"
        _mathtext_tt = "Arial"
    elif font_case_insensitive == "georgia":
        _fontfamily = "Georgia"
        _mathtext_fontset = "stix"
        _mathtext_tt = "Georgia"
    elif font_case_insensitive == "verdana":
        _fontfamily = "Verdana"
        _mathtext_fontset = "custom"
        _mathtext_tt = "Verdana"
    elif font_case_insensitive == "courier new":
        _fontfamily = "Courier New"
        _mathtext_fontset = "custom"
        _mathtext_tt = "Courier New"
    else:
        print("Unsupported font. Please manually enter the 'font.family', 'mathtext.fontset', " + \
              "and 'mathtext.tt' attributes of matplotlib.")
        _fontfamily = input("font.family: ")
        _mathtext_fontset = input("mathtext.fontset: ")
        _mathtext_tt = input("mathtext.tt: ")
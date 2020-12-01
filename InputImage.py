from pdf2image import convert_from_path, convert_from_bytes
from Image import Image


def Main():

    # convert from pdf
    name = "map-3"
    s = 4000
    size = (s, None)
    image = convert_from_path("stock_pictures/%s.pdf" % name, dpi=300, single_file=True, use_pdftocairo=True,
                              size=size)
    i = Image("", image[0])
    # i.displayV2(size=8)
    image[0].save("/Users/matthieumerville/Desktop/Art/Test/pdf_%.0f.png" % s, "PNG")
    print(len(image))
    return None


if __name__ == "__main__":
    Main()

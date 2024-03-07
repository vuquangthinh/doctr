


def get_characters_from_file(file_path):
    with open(file_path, 'r') as file:
        # Read the contents of the file
        text = file.read()

    # Create an empty set to store unique characters
    unique_chars = set()

    # Iterate over each character in the text
    for char in text:
        # Add the character to the set
        unique_chars.add(char)

    return unique_chars

# Example usage
file_path = '/Users/vuquangthinh/Documents/fencilux/eKYC/KHM_TrainOCR/doctr/traindata.txt'
characters = get_characters_from_file(file_path)

with open('./char.txt', 'w') as f:
    f.write(' '.join(characters))

vocab = 'កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអឣឤឥឦឧឨឩឪឫឬឭឮឯឰឱឲឳ឴឵ាិីឹឺុូួើឿៀេែៃោៅំះៈ៉៊់៌៍៎៏័៑្៓។៕៖ៗ៘៙៚៛ៜ៝០១២៣៤៥៦៧៨៩៰៱៲៳៴៵៶៷៸៹᧠᧡᧢᧣᧤᧥᧦᧧᧨᧩᧪᧫᧬᧭᧮᧯᧰᧱᧲᧳᧴᧵᧶᧷᧸᧹᧺᧻᧼᧽᧾᧿Ⳉⳉ' + ''.join(['˜', "'", '“', '.', ',', '"', '4', '7', '3', '6', '!', '2', '”', '»', '8', 'g', '0', 'p', '?', '&', '5', '1', '9', '«', 'j', '[', ']', '°', '~', '¦', '£', '$', '%', '€', '…', '^', '&', '×', '@', '‹', '›', '⁄', '#', '±', '‘', '_', '¥', '©', '÷', '’', ' '])

characters_not_in_y = [char for char in characters if char not in vocab]


print(characters_not_in_y)
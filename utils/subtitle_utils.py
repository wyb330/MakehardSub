from subtitle.subtitle import Subtitle
from utils.file_utils import *
from subtitle.generic import Style


def to_smi_timestamp(total_seconds):
    return str(int(total_seconds * 1000))


def to_srt_timestamp(total_seconds):
    total_seconds = total_seconds / 1000
    hours = int(total_seconds / 3600)
    minutes = int(total_seconds / 60 - hours * 60)
    seconds = int(total_seconds - hours * 3600 - minutes * 60)
    milliseconds = round((total_seconds - seconds - hours * 3600 - minutes * 60)*1000)

    return '{:02d}:{:02d}:{:02d}.{:03d}'.format(hours, minutes, seconds, milliseconds)


def str2time(time):
    t = time.split('.')
    hms = t[0].split(':')
    seconds = int(hms[0]) * 3600 + 60 * int(hms[1]) + int(hms[2]) + (int(t[1]) / 1000)
    return int(seconds * 1000)


def subtitle_captions(file):
    ext = extract_file_extension(file)
    if ext == '.srt':
        captions = Subtitle().from_srt(file)
    elif ext == '.vtt':
        captions = Subtitle().from_vtt(file)
    elif ext == '.smi' or ext == '.sami':
        try:
            captions = Subtitle().from_smi(file)
        except AttributeError:
            raise AttributeError('AttributeError - {}'.format(file))
    else:
        # raise Exception('Unknown subtitle format')
        return None

    return captions


def save_as_smi(filename, texts, times):
    smi_style = """
    <STYLE TYPE="text/css">
    <!--
    P { margin-left:8pt; margin-right:8pt; margin-bottom:2pt;
        margin-top:2pt; font-size:20pt; text-align:center;
        font-family:arial, sans-serif; font-weight:normal; color:white;
        }
    .KRCC {Name:Korean; lang:ko-KR; SAMIType:CC;}
    .ENCC {Name:English; lang:EN-US; SAMIType:CC;}
    -->
    </STYLE>
    """
    style = Style()
    style.lines = smi_style
    subtitle = Subtitle(styles=style)
    for time, text in zip(times, texts):
        subtitle.add(to_srt_timestamp(time[0]), to_srt_timestamp(time[1]), text)

    subtitle.save_as_smi(filename)


def save_to_srt(filename, texts, times):
    subtitle = Subtitle()
    for time, text in zip(times, texts):
        if text == "&nbsp;":
            continue
        subtitle.add(to_srt_timestamp(time[0]), to_srt_timestamp(time[1]), text)

    subtitle.save_as_srt(filename)


def save_as_srt(filename, captions):
    subtitle = Subtitle()
    for caption in captions:
        if caption.text == "&nbsp;":
            continue
        subtitle.add(caption.start, caption.end, caption.text)

    subtitle.save_as_srt(filename)

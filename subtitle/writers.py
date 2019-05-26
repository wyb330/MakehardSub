from subtitle.generic import Style


class WebVTTWriter(object):

    def write(self, captions, f):
        f.write('WEBVTT\n')
        for c in captions:
            f.write('\n{} --> {}\n'.format(c.start, c.end))
            f.writelines(['{}\n'.format(l) for l in c.lines])


class SRTWriter(object):

    def write(self, captions, f):
        for line_number, caption in enumerate(captions, start=1):
            f.write('{}\n'.format(line_number))
            f.write('{} --> {}\n'.format(self._to_srt_timestamp(caption.start_in_seconds),
                                         self._to_srt_timestamp(caption.end_in_seconds)))
            f.writelines(['{}\n'.format(l) for l in caption.lines])
            f.write('\n')

    def _to_srt_timestamp(self, total_seconds):
        hours = int(total_seconds / 3600)
        minutes = int(total_seconds / 60 - hours * 60)
        seconds = int(total_seconds - hours * 3600 - minutes * 60)
        milliseconds = round((total_seconds - seconds - hours * 3600 - minutes * 60)*1000)

        return '{:02d}:{:02d}:{:02d},{:03d}'.format(hours, minutes, seconds, milliseconds)


class SBVWriter(object):
    pass


class SMIWriter(object):

    def __init__(self, title=None, style=None):
        self.title = title
        self.style = style

    def _write_header(self, f):
        f.write('<SAMI>\n')
        f.write('\t<HEAD>\n')
        if self.title:
            f.write('\t\t<TLTLE>{}</TITLE>\n'.format(self.title))
        self._write_style(f)
        f.write('\t</HEAD>\n')
        f.write('\t<BODY>\n')

    def _write_style(self, f):
        if type(self.style) == Style:
            for line in self.style.lines:
                f.write(line)

    def _write_body(self, f, captions):
        for line_number, caption in enumerate(captions, start=1):
            if caption.identifier is None:
                p_class = 'KRCC'
            else:
                p_class = caption.identifier
            f.write('\t\t<SYNC Start={}> <P class={}>'.format(self._to_timestamp(caption.start), p_class))
            f.write(caption.raw_text + '\n')

    def _write_footer(self, f):
        f.write('\t</BODY>\n')
        f.write('</SAMI>\n')

    def write(self, captions, f):
        self._write_header(f)
        self._write_body(f, captions)
        self._write_footer(f)

    def _to_timestamp(self, time):
        """ time format: '{:02d}:{:02d}:{:06.3f} """
        t = time.split('.')
        hms = t[0].split(':')
        seconds = int(hms[0]) * 3600 + 60 * int(hms[1]) + int(hms[2]) + int(t[1]) / 1000
        return int(seconds * 1000)

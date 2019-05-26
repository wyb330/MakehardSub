import re
import os
import codecs

from subtitle.exceptions import MalformedFileError, MalformedCaptionError
from subtitle.generic import GenericParser, Caption, Block, Style


class TextBasedParser(GenericParser):
    """
    Parser for plain text caption files.
    This is a generic class, do not use directly.
    """

    TIMEFRAME_LINE_PATTERN = ''
    PARSER_OPTIONS = {}

    def _read_content(self, file):

        first_bytes = min(32, os.path.getsize(file))
        with open(file, 'rb') as f:
            raw = f.read(first_bytes)

        skip = 0
        if raw.startswith(codecs.BOM_UTF8):
            encoding = 'utf-8-sig'
        elif raw.startswith(codecs.BOM_UTF16_LE):
            encoding = 'utf-16-le'
            skip = 2
        elif raw.startswith(codecs.BOM_UTF16_BE):
            encoding = 'utf-16-be'
            skip = 2
        else:
            encoding = 'utf-8'

        try:
            with open(file, encoding=encoding) as f:
                lines = [line.rstrip('\n') for line in f.readlines()]
            if encoding in ['utf-16-le', 'utf-16-be']:
                lines[0] = lines[0][skip-1:]
        except UnicodeDecodeError:
            with open(file, errors='ignore') as f:
                lines = [line.rstrip('\n') for line in f.readlines()]

        if not lines:
            raise MalformedFileError('The file is empty.')

        return lines

    def _parse_timeframe_line(self, line):
        """Parse timeframe line and return start and end timestamps."""
        tf = self._validate_timeframe_line(line)
        if not tf:
            raise MalformedCaptionError('Invalid time format')

        return tf.group(1), tf.group(2)

    def _validate_timeframe_line(self, line):
        return re.match(self.TIMEFRAME_LINE_PATTERN, line)

    def _is_timeframe_line(self, line):
        """
        This method returns True if the line contains the timeframes.
        To be implemented by child classes.
        """
        return False

    def _should_skip_line(self, line, index, caption):
        """
        This method returns True for a line that should be skipped.
        To be implemented by child classes.
        """
        return False

    def _parse(self, lines):
        c = None

        for index, line in enumerate(lines):
            line = line.strip()
            if self._is_timeframe_line(line):
                try:
                    start, end = self._parse_timeframe_line(line)
                except MalformedCaptionError as e:
                    raise MalformedCaptionError('{} in line {}'.format(e, index + 1))
                c = Caption(start, end)
            elif self._should_skip_line(line, index, c):  # allow child classes to skip lines based on the content
                continue
            elif line:
                if c is None:
                    raise MalformedCaptionError(
                        'Caption missing timeframe in line {}.'.format(index + 1))
                else:
                    c.add_line(line)
            else:
                if c is None:
                    continue
                if not c.lines:
                    if self.PARSER_OPTIONS.get('ignore_empty_captions', False):
                        c = None
                        continue
                    raise MalformedCaptionError('Caption missing text in line {}.'.format(index + 1))

                self.captions.append(c)
                c = None

        if c is not None and c.lines:
            self.captions.append(c)


class SRTParser(TextBasedParser):
    """
    SRT parser.
    """

    TIMEFRAME_LINE_PATTERN = re.compile('\s*(\d+:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d+:\d{2}:\d{2}[,.]\d{1,3})')

    PARSER_OPTIONS = {
        'ignore_empty_captions': True
    }

    def _validate(self, lines):
        if len(lines) < 2 or not lines[0].isdigit() or not self._validate_timeframe_line(lines[1]):
            raise MalformedFileError('The file does not have a valid format.')

    def _is_timeframe_line(self, line):
        return '-->' in line

    def _should_skip_line(self, line, index, caption):
        return caption is None and line.isdigit()


class WebVTTParser(TextBasedParser):
    """
    WebVTT parser.
    """

    TIMEFRAME_LINE_PATTERN = re.compile('\s*((?:\d+:)?\d{2}:\d{2}.\d{3})\s*-->\s*((?:\d+:)?\d{2}:\d{2}.\d{3})')
    COMMENT_PATTERN = re.compile('NOTE(?:\s.+|$)')
    STYLE_PATTERN = re.compile('STYLE[ \t]*$')

    def __init__(self):
        super().__init__()
        self.styles = []

    def _compute_blocks(self, lines):
        blocks = []

        for index, line in enumerate(lines, start=1):
            if line:
                if not blocks:
                    blocks.append(Block(index))
                if not blocks[-1].lines:
                    blocks[-1].line_number = index
                blocks[-1].lines.append(line)
            else:
                blocks.append(Block(index))

        # filter out empty blocks and skip signature
        self.blocks = list(filter(lambda x: x.lines, blocks))[1:]

    def _parse_cue_block(self, block):
        caption = Caption()
        cue_timings = None

        for line_number, line in enumerate(block.lines):
            if self._is_cue_timings_line(line):
                if cue_timings is None:
                    try:
                        cue_timings = self._parse_timeframe_line(line)
                    except MalformedCaptionError as e:
                        raise MalformedCaptionError(
                            '{} in line {}'.format(e, block.line_number + line_number))
                else:
                    raise MalformedCaptionError(
                        '--> found in line {}'.format(block.line_number + line_number))
            elif line_number == 0:
                caption.identifier = line
            else:
                caption.add_line(line)

        caption.start = cue_timings[0]
        caption.end = cue_timings[1]
        return caption

    def _parse(self, lines):
        self._compute_blocks(lines)

        for block in self.blocks:
            if self._is_cue_block(block):
                caption = self._parse_cue_block(block)
                self.captions.append(caption)
            elif self._is_comment_block(block):
                continue
            elif self._is_style_block(block):
                if self.captions:
                    raise MalformedFileError(
                        'Style block defined after the first cue in line {}.'
                        .format(block.line_number))
                style = Style()
                style.lines = block.lines[1:]
                self.styles.append(style)
            else:
                if len(block.lines) == 1:
                    raise MalformedCaptionError(
                        'Standalone cue identifier in line {}.'.format(block.line_number))
                else:
                    raise MalformedCaptionError(
                        'Missing timing cue in line {}.'.format(block.line_number+1))

    def _validate(self, lines):
        if not re.match('WEBVTT', lines[0]):
            raise MalformedFileError('The file does not have a valid format')


    def _is_cue_timings_line(self, line):
        return '-->' in line

    def _is_cue_block(self, block):
        """Returns True if it is a cue block
        (one of the two first lines being a cue timing line)"""
        return any(map(self._is_cue_timings_line, block.lines[:2]))

    def _is_comment_block(self, block):
        """Returns True if it is a comment block"""
        return re.match(self.COMMENT_PATTERN, block.lines[0])

    def _is_style_block(self, block):
        """Returns True if it is a style block"""
        return re.match(self.STYLE_PATTERN, block.lines[0])


class SBVParser(TextBasedParser):
    """
    YouTube SBV parser.
    """

    TIMEFRAME_LINE_PATTERN = re.compile('\s*(\d+:\d{2}:\d{2}.\d{3}),(\d+:\d{2}:\d{2}.\d{3})')

    def _validate(self, lines):
        if not self._validate_timeframe_line(lines[0]):
            raise MalformedFileError('The file does not have a valid format')

    def _is_timeframe_line(self, line):
        return self._validate_timeframe_line(line)


def _tplit(s, tag):
    delimiter = '<' + tag
    try:
        return [(delimiter + item).strip() for item in re.split(delimiter, s, flags=re.I)][1:]
    except:
        return []


def _lookup(s, pattern):
    return re.search(pattern, s, flags=re.I)


def _normalize(content):
    content = content.replace('\n', ' ')
    content = re.sub('<br ?/?>', '\n', content, flags=re.I)
    content = re.sub('<.*?>', '', content)
    content = content.replace('&nbsp', '')
    content = content.strip()
    return content


def _plang(item):
    try:
        lang = _lookup(item, '<p(.+)class=([a-z]+)').group(2)
    except AttributeError:
        raise AttributeError('AttributeError - {}'.format(item))
    content = item[_lookup(item, '<p([^>]+)>').end():]
    content = content.replace('\n', ' ')
    content = re.sub('<br ?/?>', '\n', content, flags=re.I)
    content = re.sub('<.*?>', '', content)
    content = content.strip()
    return [lang, content]


class SMIParser(TextBasedParser):
    """
    SMI parser.
    """

    TIMEFRAME_LINE_PATTERN = '(<SYNC Start=)(\d+)>'
    P_TAG_PATTERN = '\s*(\d+)\s*'

    PARSER_OPTIONS = {
        'ignore_empty_captions': True
    }

    def __init__(self):
        super().__init__()
        self.styles = []

    def _validate(self, lines):
        if len(lines) < 2 or (lines[0].upper() != '<SAMI>' and not lines[0].startswith('<!--')):
            raise MalformedFileError('The file does not have a valid format.')

    def _is_timeframe_line(self, line):
        return re.findall(self.TIMEFRAME_LINE_PATTERN, line)

    def _should_skip_line(self, line, index, caption):
        return caption is None and line.isdigit()

    def _parse(self, lines):
        raw_text = '\n'.join(lines)
        for index, item in enumerate(_tplit(raw_text, 'sync')):
            timecode = Caption().to_timestamp(int(_lookup(item, '<sync start=([0-9]+)').group(1)) / 1000)
            content = dict(map(_plang, _tplit(item, 'p')))
            if len(content.keys()) > 0:
                lang = list(content.keys())[0]
                caption = Caption(start=timecode, text=content[lang])
            else:
                content = _normalize(item)
                caption = Caption(start=timecode, text=content)
            if index > 0:
                self.captions[-1].end = timecode
            self.captions.append(caption)

#!/usr/bin/env python
# -*- coding: UTF-8 -*-

class Color (object):

  # It's strict on what to accept, but I kinda like it that way.
  def __init__(self, r=0, g=0, b=0):
    self.r = r
    self.g = g
    self.b = b

  # Maybe this would be a better __init__?
  # The first may be more clear but this could handle way more cases...
  # I like the first more though. What do you think?
  #
  #def __init__(self, obj):
  #  self.r, self.g, self.b = list(obj)[:3]

  # This methods allows to use lists longer than 3 items (eg. rgba), where
  # 'Color(*alist)' would bail
  @classmethod
  def from_List(cls, alist):
    r, g, b = alist[:3]
    return cls(r, g, b)

  # So we could use dicts with more keys than rgb keys, where
  # 'Color(**adict)' would bail
  @classmethod
  def from_Dict(cls, adict):
    return cls(adict['r'], adict['g'], adict['b'])

  # This should theoreticaly work with every object that's iterable.
  # Maybe that's more intuitive duck typing than to rely on an object
  # to have an as_List() methode or similar.
  @classmethod
  def from_Object(cls, obj):
    return cls.from_List(list(obj))

  def __str__(self):
    return "<Color(%s, %s, %s)>" % (self.r, self.g, self.b)

  def _set_rgb(self, r, g, b):
    self.r = r
    self.g = g
    self.b = b
  def _get_rgb(self):
    return  (self.r, self.g, self.b)
  rgb = property(_get_rgb, _set_rgb)

  def as_List(self):
    return [self.r, self.g, self.b]

  def __iter__(self):
    return (c for c in (self.r, self.g, self.b))

  # We could add a single value (to all colorvalues) or a list of three
  # (or more) values (from any object supporting the iterator protocoll)
  # one for each colorvalue
  def __add__(self, obj):
    r, g, b = self.r, self.g, self.b
    try:
      ra, ga, ba = list(obj)[:3]
    except TypeError:
      ra = ga = ba = obj
    r += ra
    g += ga
    b += ba
    return Color(*Color.check_rgb(r, g, b))

  @staticmethod
  def check_rgb(*vals):
    ret = []
    for c in vals:
      c = int(c)
      c = min(c, 255)
      c = max(c, 0)
      ret.append(c)
    return ret

class ColorAlpha(Color):

  def __init__(self, r=0, g=0, b=0, alpha=255):
    Color.__init__(self, r, g, b)
    self.alpha = alpha

  def __str__(self):
    return "<Color(%s, %s, %s, %s)>" % (self.r, self.g, self.b, self.alpha)

  # ...

if __name__ == '__main__':
  l = (220, 0, 70)
  la = (57, 58, 61, 255)
  d = {'r': 220, 'g': 0, 'b':70}
  da = {'r': 57, 'g': 58, 'b':61, 'a':255}
  c = Color(); print c # <Color(0, 0, 0)>
  ca = ColorAlpha(*la); print ca # <Color(57, 58, 61, 255)>
  print '---'
  c = Color(220, 0, 70); print c # <Color(220, 0, 70)>
  c = Color(*l); print c # <Color(220, 0, 70)>
  #c = Color(*la); print c # -> Fail
  c = Color(**d); print c # <Color(220, 0, 70)>
  #c = Color(**da); print c # -> Fail
  print '---'
  c = Color.from_Object(c); print c # <Color(220, 0, 70)>
  c = Color.from_Object(ca); print c # <Color(57, 58, 61, 255)>
  c = Color.from_List(l); print c # <Color(220, 0, 70)>
  c = Color.from_List(la); print c # <Color(57, 58, 61, 255)>
  c = Color.from_Dict(d); print c # <Color(220, 0, 70)>
  c = Color.from_Dict(da); print c # <Color(57, 58, 61, 255)>
  print '---'
  print 'Check =', Color.check_rgb('1', 0x29a, -23, 40)
  # Check = [1, 255, 0, 40]
  print '%s + %s = %s' % (c, 10, c + 10)
  # <Color(57, 58, 61)> + 10 = <Color(67, 68, 71)>
  print '%s + %s = %s' % (c, ca, c + ca)
  # <Color(57, 58, 61)> + <Color(57, 58, 61, 255)> = <Color(114, 116, 122)>


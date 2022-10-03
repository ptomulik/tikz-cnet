#
# Copyright (c) 2022 by Pawel Tomulik <ptomulik@wp.pl>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

import os

env = Environment(ENV={"PATH": os.environ["PATH"]})

env['BUILDERS']['PGFGen'] = Builder(
  action = '$PGFGEN $_PGFGEN_TEMPLATE_PATH_FLAGS $_PGFGEN_SVG_PATH_FLAGS -o $TARGET $SOURCE',
  suffix = '.tex',
  src_suffix = '.tex.jinja'
)
env.SetDefault(
    PGFGEN='pgfgen',
    PGFGEN_TEMPLATE_PATH_PREFIX="-T ",
    PGFGEN_TEMPLATE_PATH_SUFFIX="",
    PGFGEN_TEMPLATE_PATH=[],
    PGFGEN_SVG_PATH_PREFIX="-S ",
    PGFGEN_SVG_PATH_SUFFIX="",
    PGFGEN_SVG_PATH=[],
    _PGFGEN_TEMPLATE_PATH_FLAGS='${_concat(PGFGEN_TEMPLATE_PATH_PREFIX, PGFGEN_TEMPLATE_PATH, PGFGEN_TEMPLATE_PATH_SUFFIX, __env__, RDirs, TARGET, SOURCE, affect_signature=False)}',
    _PGFGEN_SVG_PATH_FLAGS='${_concat(PGFGEN_SVG_PATH_PREFIX, PGFGEN_SVG_PATH, PGFGEN_SVG_PATH_SUFFIX, __env__, RDirs, TARGET, SOURCE, affect_signature=False)}'
)

env.Append(PGFGEN_TEMPLATE_PATH=[env.Dir('.'), env.Dir('src/templates')])
env.Append(PGFGEN_SVG_PATH=[env.Dir('src/svg')])

VariantDir(variant_dir = 'build', src_dir = 'src', duplicate = 1)
SConscript('build/SConscript', exports = ['env'])

# Local Variables:
# # tab-width:4
# # indent-tabs-mode:nil
# # End:
# vim: set syntax=python expandtab tabstop=4 shiftwidth=4:

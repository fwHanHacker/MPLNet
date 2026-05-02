import glob
import os
import re
from setuptools import find_packages, setup
try:
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
except Exception:
    CUDAExtension = None
    BuildExtension = None


def parse_requirements(fname='requirements.txt', with_version=True):
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        if line.startswith('-r '):
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


install_requires = parse_requirements()


def get_extensions():
    extensions = []

    op_files = glob.glob('./mfplnet/ops/csrc/*.c*')
    if CUDAExtension is None or not op_files:
        return extensions
    extension = CUDAExtension
    ext_name = 'mfplnet.ops.nms_impl'

    ext_ops = extension(
        name=ext_name,
        sources=op_files,
    )

    extensions.append(ext_ops)

    return extensions


cmdclass = {'build_ext': BuildExtension} if BuildExtension is not None and get_extensions() else {}
ext_modules = get_extensions()


setup(name='mfplnet',
      version="1.0.0",
      description='Official implementation of MPLNet for power-line instance detection',
      keywords='computer vision & lane detection',
      classifiers=[
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Intended Audience :: Developers',
          'Operating System :: OS Independent'
      ],
      packages=find_packages(),
      include_package_data=True,
      python_requires='>=3.8',
      install_requires=install_requires,
      ext_modules=ext_modules,
      cmdclass=cmdclass,
      zip_safe=False)

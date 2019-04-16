MAKING TENSORFLOW WORK WITH AIMSUN NEXT 8.2.2:
- Trying to import tensorflow from a python script that uses the Aimsun API does not work
- The problem comes from the hdf5 library. Aimsun uses an older version internally (packaged with the Aimsun Next distribution)
- Tensorflow imports h5py, which upon installation looks up for the system's hdf5, which will likely be a newer version
- When import tensorflow from Aimsun then raises an exception saying that the hdf5 of the application does not match the one from the library
- The solution was to manually download the correct (older) version from the hdf5 website, then re-install h5py from pip pointing to the location of the downloaded (older) hdf5 library:
cd ~
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.11/bin/linux-x86_64/hdf5-1.8.11-linux-x86_64-shared.tar.gz
tar -xvzf hdf5-1.8.11-linux-x86_64-shared.tar.gz
sudo pip uninstall h5py
export LD_LIBRARY_PATH=/home/rodr/hdf5-1.8.11-linux-x86_64-shared/lib:/usr/local/cuda/lib64:
sudo HDF5_VERSION=1.8.11 HDF5_DIR=/home/rodr/hdf5-1.8.11-linux-x86_64-shared pip install --no-binary=h5py h5py

You can check the installed hdf5 and h5py versions using:
python -c 'import h5py; print(h5py.version.info)'

The version of hdf5 returned must match the one used internally by Aimsun, which for Aimsun 8.2.2 is HDF5 1.8.11.


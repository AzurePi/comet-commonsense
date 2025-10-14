mkdir -p data/atomic
cd data/atomic

wget https://storage.googleapis.com/ai2-mosaic/public/atomic/v1.0/atomic_data.tgz

tar -xvzf atomic_data.tgz
rm atomic_data.tgz

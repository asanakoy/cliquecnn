declare -a categories=(
 'basketball_layup'
 'bowling'
 'clean_and_jerk'
 'discus_throw'
 'diving_platform_10m'
 'diving_springboard_3m'
 'hammer_throw'
 'high_jump'
 'javelin_throw'
 'long_jump'
 'pole_vault'
 'shot_put'
 'snatch'
 'tennis_serve'
 'triple_jump'
 'vault')

url_prefix="https://hcicloud.iwr.uni-heidelberg.de/index.php/s/kRp6b454Dd0wnts/download?path=%2F&files="
for category in "${categories[@]}"
do
    url="${url_prefix}${category}.caffemodel"
    wget $url -O "${category}.caffemodel"
done

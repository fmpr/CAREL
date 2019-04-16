TARGET=$1
kill $(ps aux | grep 'aconsole' | grep $TARGET | awk '{print $2}')

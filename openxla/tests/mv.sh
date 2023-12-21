while IFS= read -r file; do
    echo $file
    mv "$file" ../blacklist
done < ../blacklist.txt

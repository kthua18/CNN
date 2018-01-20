Hello Pumpkim Hua,
Here are the basic "happy path" steps to using git:

* To get a git repository onto your computer, go to the repository's page on Github then click the green
  "clone" button and copy the url. In your terminal, type `git clone <url>` to download it.
* After you've modified the repo (by adding or editing files), you must _stage_ your changes. This is accomplished
  by typing `git add <filename>` in the terminal. If you want to just add everything, you can type `git add -A`
* You must commit your changes using `git commit`. Commiting basically means that you have finished a "batch" of
  changes. A commit becomes an entry in the history log of your repo.
  When you type `git commit`, a text editor will open automatically for you to write a message to
  describe your commit, because you _must_ attach a message to your commit. (when done, type ESC :wq)
  
  You can write a message and commit at the
  same time by typing `git commit -m 'this is my commit message'`.
* To sync your new commit(s) to Github, you type `git push` then type your Github credentials in the prompt.
* To fetch the latest changes and synchronize your repo with Github, type `git pull`.

Yay :)  :>

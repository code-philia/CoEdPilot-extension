function update(){
    if (pterosaurCleanupCheck !== useFullVim())
      cleanupPterosaur();
    if (debugMode && !options["pterosaurdebug"])
    {
      killVimbed();
      startVimbed(0);
    }
    else if (!debugMode && options["pterosaurdebug"])
    {
      killVimbed();
      startVimbed(1);
    }
    var cursorPos;
    if(dactyl.focusedElement === pterFocused && textBoxType)
    {
      cursorPos = textBoxGetSelection()
      if (savedCursorStart!=null &&
           (cursorPos.start.row != savedCursorStart.row || cursorPos.start.column != savedCursorStart.column) ||
          savedCursorEnd!=null &&
           (cursorPos.end.row != savedCursorEnd.row || cursorPos.end.column != savedCursorEnd.column))
      {
        updateTextbox(0);
        return;
      }
      if (savedText!=null && textBoxGetValue() != savedText)
      {
        updateTextbox(1);
        return;
      }
    }
    //This has to be up here for vimdo to work. This should probably be changed eventually.
    if (writeInsteadOfRead)
    {
      if(sendToVim !== "")
      {
        let tempSendToVim=sendToVim
        sendToVim = ""
        io.system("printf '" + tempSendToVim  + "' > /tmp/vimbed/pterosaur_"+uid+"/fifo");
        unsent=0;
        cyclesSinceLastSend=0;
      }

      if(cyclesSinceLastSend < 2)
      {
        io.system('vim --servername pterosaur_'+uid+' --remote-expr "Vimbed_Poll()" &');
        cyclesSinceLastSend+=1;
      }
      writeInsteadOfRead = 0;
      return;
    }
    // More ...
}
function updateVim(){
    if(sendToVim !== "")
    {
      let tempSendToVim=sendToVim
      sendToVim = ""
      io.system("printf '" + tempSendToVim  + "' > /tmp/vimbed/pterosaur_"+uid+"/fifo");
      unsent=0;
      cyclesSinceLastSend=0;
    }
}
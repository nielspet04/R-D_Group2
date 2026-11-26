// Local mode remains dependency-free and never initializes cloud authentication.
window.monitorSource = {
  mode:'local', ready:Promise.resolve(),
  async request(url){const response=await fetch(url);if(!response.ok)throw Error('HTTP '+response.status);return response.json()}
};

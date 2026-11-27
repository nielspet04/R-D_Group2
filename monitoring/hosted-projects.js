// Project discovery uses the signed-in user's RLS permissions, never a server key.
export class ProjectScope {
  constructor(preferredId) { this.preferredId=preferredId; this.reset(); }
  reset(userId=null) { this.version=(this.version||0)+1; this.userId=userId; this.projectId=null; this.projects=[]; }
  async discover(client) {
    const version=this.version;
    const {data,error}=await client.from('monitor_projects').select('id,name').order('name');
    if(version!==this.version)return false;
    if(error)throw Error('Could not load your projects. Try signing in again.');
    this.projects=data||[];
    this.projectId=this.projects.find(p=>p.id===this.preferredId)?.id||this.projects[0]?.id||null;
    return true;
  }
  choose(id) {
    if(!this.projects.some(p=>p.id===id))throw Error('Project access required.');
    this.version++; this.projectId=id;
  }
  capture() {
    if(!this.userId)throw Error('Sign in to view private results.');
    if(!this.projectId)throw Error('Your account has no project access yet. Ask the dashboard administrator to add your User ID.');
    return {version:this.version,projectId:this.projectId};
  }
  assert(context) {
    if(context.version!==this.version||context.projectId!==this.projectId)throw Error('Account or project changed; response discarded.');
  }
}

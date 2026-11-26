import {build} from 'esbuild';
import {mkdir,readFile,writeFile,copyFile} from 'node:fs/promises';
const directory='dist';await mkdir(directory,{recursive:true});
const configPath=process.env.MONITOR_PUBLIC_CONFIG||'configs/monitoring-hosted.example.json';
const config=JSON.parse(await readFile(configPath,'utf8'));
const allowed=['supabaseUrl','publishableKey','projectId'];
if(Object.keys(config).some(k=>!allowed.includes(k)))throw Error('Unknown or secret configuration field');
if(config.publishableKey&&(!config.publishableKey.startsWith('sb_publishable_')&&!config.publishableKey.startsWith('eyJ')))throw Error('Use a publishable/anon key, never a secret key');
if(config.publishableKey?.startsWith('eyJ')){const claims=JSON.parse(Buffer.from(config.publishableKey.split('.')[1],'base64url').toString());if(claims.role!=='anon')throw Error('Privileged JWT cannot be built into website')}
let html=await readFile('monitoring/index.html','utf8');
html=html.replace('<script src="/data-source.js" defer></script>','<script src="/config.js" defer></script><script src="/hosted-source.js" defer></script>');
html=html.replace('</head>','<link rel="manifest" href="/manifest.webmanifest"><meta name="theme-color" content="#0c121c"></head>');
await writeFile(directory+'/index.html',html);
await writeFile(directory+'/config.js','window.MONITOR_CONFIG='+JSON.stringify(config)+';\n');
for(const file of ['dashboard.css','dashboard.js','manifest.webmanifest','icon.svg'])await copyFile('monitoring/'+file,directory+'/'+file);
await build({entryPoints:['monitoring/hosted-entry.js'],outfile:directory+'/hosted-source.js',bundle:true,minify:true,platform:'browser',target:['es2022'],legalComments:'eof'});
const connect=config.supabaseUrl||'https://*.supabase.co';
await writeFile(directory+'/_headers',`/*\n  X-Content-Type-Options: nosniff\n  Referrer-Policy: no-referrer\n  Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self' ${connect}; img-src 'self' data:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'\n/config.js\n  Cache-Control: no-store\n`);
console.log('Built dist/: static website with public configuration only.');

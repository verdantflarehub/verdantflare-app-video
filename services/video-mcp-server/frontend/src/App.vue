<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

type Task = {
  video_task_id: string
  project_id: string
  service: string
  route?: string
  model?: string
  prompt?: string
  status: string
  created_at: string
  completed_at?: string | null
  duration_seconds?: number | null
  aspect_ratio?: string | null
  media?: { width?: number; height?: number; frame_rate?: number }
  timing?: { total_seconds?: number | null }
}

const base = location.pathname.split('/dashboard')[0]
const tokenKey = `verdantflare.video.dashboard.token:${base}`
const token = ref(localStorage.getItem(tokenKey) || '')
const connected = ref(false)
const loading = ref(false)
const error = ref('')
const active = ref<'tasks' | 'models' | 'mcp'>('tasks')
const tasks = ref<Task[]>([])
const query = ref('')
const status = ref('all')
const service = ref('all')

const labels: Record<string, string> = { queued: '排队中', running: '渲染中', succeeded: '已完成', failed: '失败', cancelled: '已取消' }
const generation = new Set(['h3', 'h3-sol', 'h3-vdn'])
const operations: Record<string, string> = { h3: '视频生成', 'h3-sol': '视频生成', 'h3-vdn': '视频生成', 'h3-latent-upscale': 'H3 潜空间处理', sr: '视频超分', interpolate: '视频补帧', depth: '深度视频' }
const services: Record<string, string> = { 'h3-latent-upscale': 'H3 Latent Upscaler', sr: 'Video Super Resolution', interpolate: 'Frame Interpolation', depth: 'Depth Anything' }
const filtered = computed(() => tasks.value.filter((task) => {
  const text = `${task.video_task_id} ${task.project_id} ${task.prompt || ''}`.toLowerCase()
  return (status.value === 'all' || task.status === status.value) && (service.value === 'all' || task.service === service.value) && (!query.value || text.includes(query.value.toLowerCase()))
}))
const counts = computed(() => Object.fromEntries(['queued', 'running', 'succeeded', 'failed', 'cancelled'].map((key) => [key, tasks.value.filter((task) => task.status === key).length])))
const modelName = (task: Task) => task.model || (generation.has(task.service) ? 'minimax-h3-ref2va' : '')
const routeName = (task: Task) => generation.has(task.service) && ['h3', 'h3-sol', 'h3-vdn'].includes(task.route || task.service) ? (task.route || task.service) : ''
const identity = (task: Task) => generation.has(task.service) ? [`模型：${modelName(task)}`, routeName(task) ? `渠道：${routeName(task)}` : ''] : [`操作：${operations[task.service] || '视频处理'}`, `服务：${services[task.service] || task.service || '未指定'}`]
const date = (value: string) => new Date(value).toLocaleString()
const duration = (task: Task) => task.timing?.total_seconds != null ? `${Math.floor(task.timing.total_seconds)}s` : '—'

async function load() {
  if (!token.value) return
  loading.value = true; error.value = ''
  try {
    const response = await fetch(`${base}/api/dashboard?page_size=100`, { headers: { Authorization: `Bearer ${token.value}` } })
    if (!response.ok) throw new Error(response.status === 401 ? 'Token 无效或已过期' : `请求失败（${response.status}）`)
    tasks.value = (await response.json()).tasks || []; connected.value = true
  } catch (cause) { connected.value = false; error.value = cause instanceof Error ? cause.message : '连接失败' }
  finally { loading.value = false }
}
function saveToken() { localStorage.setItem(tokenKey, token.value); load() }
function clearToken() { token.value = ''; localStorage.removeItem(tokenKey); connected.value = false; tasks.value = [] }
onMounted(load)
</script>

<template>
  <header class="site-header"><div class="nav"><a class="brand" href="#"><span class="brand-mark" /><span>VerdantFlare</span><span class="brand-tag">VIDEO MCP</span></a><div class="nav-actions"><span class="pill-badge">{{ connected ? '已连接' : '等待连接' }}</span><button class="button" @click="token ? clearToken() : undefined">{{ token ? '清除 Token' : '未配置 Token' }}</button></div></div></header>
  <div class="workspace"><aside class="workspace-nav" aria-label="业务导航"><p>WORKSPACE</p><button v-for="item in [['tasks','01 / TASKS'],['models','02 / MODELS'],['mcp','03 / MCP']]" :key="item[0]" class="nav-link" :class="{ selected: active === item[0] }" @click="active = item[0] as typeof active">{{ item[1] }}</button></aside>
    <main class="market-page">
      <section v-if="!token" class="market-wrap business-section"><h1>连接 Video MCP</h1><p class="business-muted">使用工作区 Token 读取真实任务、模型和服务状态。</p><form class="token-form" @submit.prevent="saveToken"><input data-testid="token-input" v-model="token" type="password" autocomplete="off" placeholder="Video MCP Token" required /><button data-testid="connect-token" class="button primary">连接</button></form></section>
      <section v-else-if="active === 'tasks'" data-testid="tasks-view" class="market-wrap business-section"><h1>任务</h1><p class="business-muted">视频生成与处理任务 · {{ filtered.length }} 条</p><div class="market-toolbar"><label class="market-search"><span>任务 / 项目 / Prompt</span><input data-testid="task-search" v-model="query" type="search" placeholder="搜索任务编号、提示词或项目…" /></label><label><span>执行服务</span><select data-testid="service-filter" v-model="service"><option value="all">全部服务</option><option v-for="item in [...new Set(tasks.map((task) => task.service))]" :key="item" :value="item">{{ services[item] || item }}</option></select></label></div><div class="market-filter-row"><button v-for="item in [['all','全部状态'],['queued','⏳ 排队中'],['running','⚡ 渲染中'],['succeeded','✓ 已完成'],['failed','✕ 失败'],['cancelled','已取消']]" :key="item[0]" :class="{ active: status === item[0] }" @click="status = item[0]">{{ item[1] }} <span v-if="item[0] !== 'all'">{{ counts[item[0]] || 0 }}</span></button></div><p v-if="loading" class="empty">正在读取任务…</p><p v-else-if="error" class="empty">{{ error }} <button class="button" @click="load">重试</button></p><p v-else-if="!filtered.length" class="empty">暂无符合条件的任务。</p><div v-else class="model-results grid"><article v-for="task in filtered" :key="task.video_task_id" data-testid="task-card" class="model-card"><div class="card-thumb-wrap"><div class="task-identity"><div v-for="line in identity(task)" v-show="line" :key="line"><span>{{ line.split('：')[0] }}：</span><strong>{{ line.split('：').slice(1).join('：') }}</strong></div></div><span class="card-badges"><span class="badge-status" :class="task.status">{{ labels[task.status] || task.status }}</span></span><div class="card-times"><span>{{ date(task.created_at) }}</span><span>总耗时 {{ duration(task) }}</span></div></div><div class="card-body"><div><div class="card-title-row">{{ task.project_id }}</div><p class="card-prompt">{{ generation.has(task.service) ? (task.prompt || '未提供提示词') : operations[task.service] || '视频处理' }}</p></div><div class="card-footer"><div class="model-tags"><span v-if="generation.has(task.service)">{{ task.aspect_ratio || '—' }}</span><span v-if="task.duration_seconds">{{ task.duration_seconds }}s</span><span v-if="task.media?.frame_rate">{{ task.media.frame_rate }} FPS</span><span v-if="task.media?.width && task.media?.height">{{ task.media.width }}×{{ task.media.height }}</span></div><span class="card-footer-action">查看详情 →</span></div></div></article></div></section>
      <section v-else-if="active === 'models'" data-testid="models-view" class="market-wrap business-section"><h1>模型服务</h1><p class="business-muted">模型、业务能力与实际执行渠道分开显示。</p><div class="business-models"><div class="business-card"><h3>minimax-h3-ref2va</h3><p>视频生成 · h3 / h3-sol / h3-vdn</p></div><div class="business-card"><h3>Video Depth Anything</h3><p>深度视频处理</p></div><div class="business-card"><h3>SeedVR2 / RIFE</h3><p>超分与补帧能力</p></div></div></section>
      <section v-else data-testid="mcp-view" class="market-wrap business-section"><h1>MCP 服务</h1><p class="business-muted">连接状态：{{ connected ? '已连接' : '未知' }}</p><div class="business-card"><h3>Video MCP Server</h3><p>任务接入、编排、Artifact 和服务观测</p></div></section>
    </main>
  </div>
</template>

<style scoped>
.token-form { display:flex; gap:12px; max-width:560px; margin-top:24px }
.token-form input { flex:1; min-width:0; border:1px solid var(--line); border-radius:8px; padding:10px 12px; background:var(--subtle); color:var(--ink) }
.task-identity { pointer-events:none }
.card-badges { position:absolute; right:12px; top:12px }
.badge-status { display:inline-flex; padding:4px 8px; border-radius:999px; background:var(--subtle); color:var(--muted); font-size:11px }
.badge-status.succeeded { color:var(--accent) }
.badge-status.failed { color:var(--danger) }
.empty { padding:42px 20px; color:var(--muted); text-align:center }
@media (max-width: 720px) { .token-form { flex-direction:column } }
</style>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import higgsfieldLogo from '../assets/higgsfield.svg?raw'
import ModelCatalog from './ModelCatalog.vue'

type Task = {
  video_task_id: string
  project_id: string
  idempotency_key?: string
  title: string
  service: string
  route?: string
  model?: string
  artifact_id?: string | null
  presentation?: { is_generation: boolean; model_label?: string; channel_label?: string; channel_icon?: string; generation_code?: string; generation_label?: string }
  prompt?: string
  status: string
  created_at: string
  completed_at?: string | null
  duration_seconds?: number | null
  aspect_ratio?: string | null
  media?: { width?: number; height?: number; frame_rate?: number; sha256?: string }
  timing?: { total_seconds?: number | null }
}

const base = location.pathname.split('/dashboard')[0]
const tokenKey = `verdantflare.video.dashboard.token:${base}`
const token = ref(localStorage.getItem(tokenKey) || '')
const connected = ref(false)
const loading = ref(false)
const loadingMore = ref(false)
const loadMoreError = ref('')
const error = ref('')
const active = ref<'tasks' | 'models' | 'mcp'>('tasks')
const tasks = ref<Task[]>([])
const query = ref('')
const status = ref('all')
const channel = ref('all')
const project = ref('all')
const channels = ref<{ id: string; label: string }[]>([])
const projects = ref<{ id: string; label: string }[]>([])
const counts = ref<Record<string, number>>({})
const total = ref(0)
const page = ref(1)
const sentinelRef = ref<HTMLElement | null>(null)
const thumbUrls = ref<Record<string, string>>({})
const thumbErrors = ref<Record<string, boolean>>({})
const thumbInFlight = new Set<string>()
let scrollObserver: IntersectionObserver | null = null

const labels: Record<string, string> = { queued: '排队中', running: '渲染中', succeeded: '已完成', failed: '失败', cancelled: '已取消' }
const generation = new Set(['h3', 'h3-sol', 'h3-vdn', 'h3-singularity', 'fal', 'grok', 'higgsfield'])
const operations: Record<string, string> = { h3: '视频生成', 'h3-sol': '视频生成', 'h3-vdn': '视频生成', fal: '视频生成', 'h3-latent-upscale': 'H3 潜空间处理', sr: '视频超分', interpolate: '视频补帧', depth: '深度视频' }
const services: Record<string, string> = { 'h3-latent-upscale': 'H3 Latent Upscaler', sr: 'Video Super Resolution', interpolate: 'Frame Interpolation', depth: 'Depth Anything' }
const filtered = computed(() => tasks.value)
const hasMore = computed(() => total.value > 0 && tasks.value.length < total.value && page.value * 24 < total.value)
const modelName = (task: Task) => task.presentation?.model_label || task.model || '未记录模型'
const routeName = (task: Task) => task.presentation?.channel_label || task.route || task.service || '未记录渠道'
const hasResultPreview = (task: Task) => task.status === 'succeeded' && Boolean(task.artifact_id || (task.service === 'higgsfield' && task.media?.sha256))
const thumbStatusText = (task: Task) => {
  if (thumbErrors.value[task.video_task_id]) return '首帧提取失败'
  if (hasResultPreview(task)) return '正在加载首帧…'
  if (task.status === 'succeeded') return '结果视频待取回'
  if (task.status === 'running') return '生成中 · 完成后提取首帧'
  if (task.status === 'queued') return '排队中 · 待生成结果'
  if (task.status === 'failed') return '任务失败 · 无结果视频'
  if (task.status === 'cancelled') return '任务已取消 · 无结果视频'
  return '暂无结果首帧'
}
const taskResolution = (task: Task) => {
  const anyTask = task as Task & { resolution?: string | null }
  if (anyTask.resolution) return anyTask.resolution
  const h = Number(task.media?.height || 0), w = Number(task.media?.width || 0)
  const shortSide = h && w ? Math.min(h, w) : 0
  if (shortSide >= 2160) return '4K'
  if (shortSide >= 1440) return '2K'
  if (shortSide >= 1080) return '1080p'
  if (shortSide >= 760) return '768p'
  if (shortSide >= 720) return '720p'
  if (shortSide >= 480) return '480p'
  const route = task.route || task.service
  if (['h3', 'h3-sol', 'h3-vdn', 'h3-singularity'].includes(route) || ['h3', 'h3-sol', 'h3-vdn', 'h3-singularity'].includes(task.service)) return '768p'
  if (route === 'fal' || task.service === 'fal') return '480p'
  if (task.service === 'higgsfield' || task.service === 'grok') return '720p'
  return ''
}
const taskFps = (task: Task) => `${task.media?.frame_rate || 24}FPS`
const isFalChannel = (task: Task) => task.presentation?.channel_icon === 'fal' || task.presentation?.channel_label === 'fal.ai' || task.route === 'fal' || task.service === 'fal'
const identity = (task: Task) => [`操作：${operations[task.service] || '视频处理'}`, `服务：${services[task.service] || task.service || '未指定'}`]
const date = (value: string) => new Date(value).toLocaleString()
const duration = (task: Task) => task.timing?.total_seconds != null ? `${Math.floor(task.timing.total_seconds)}s` : '—'

function mergeTasks(existing: Task[], incoming: Task[]) {
  const seen = new Set<string>()
  const out: Task[] = []
  for (const list of [existing, incoming]) {
    for (const item of list) {
      if (!item?.video_task_id || seen.has(item.video_task_id)) continue
      seen.add(item.video_task_id)
      out.push(item)
    }
  }
  return out
}

function revokeAllThumbnails() {
  for (const url of Object.values(thumbUrls.value)) URL.revokeObjectURL(url)
  thumbUrls.value = {}
  thumbErrors.value = {}
  thumbInFlight.clear()
}

async function fetchThumbnail(task: Task, force = false) {
  const id = task.video_task_id
  if (!token.value || !hasResultPreview(task) || document.hidden) return
  if (!force && (thumbUrls.value[id] || thumbErrors.value[id] || thumbInFlight.has(id))) return
  if (force) delete thumbErrors.value[id]
  thumbInFlight.add(id)
  try {
    const response = await fetch(`${base}/api/tasks/${encodeURIComponent(id)}/thumbnail`, {
      headers: { Authorization: `Bearer ${token.value}` },
    })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const blob = await response.blob()
    if (!blob.size) throw new Error('Empty preview')
    if (thumbUrls.value[id]) URL.revokeObjectURL(thumbUrls.value[id])
    thumbUrls.value[id] = URL.createObjectURL(blob)
    delete thumbErrors.value[id]
  } catch {
    thumbErrors.value[id] = true
  } finally {
    thumbInFlight.delete(id)
  }
}

async function loadThumbnails(list: Task[]) {
  const queue = list.filter((t) => hasResultPreview(t) && !thumbUrls.value[t.video_task_id] && !thumbErrors.value[t.video_task_id])
  let idx = 0
  async function worker() {
    while (idx < queue.length && !document.hidden) {
      const item = queue[idx++]
      await fetchThumbnail(item)
    }
  }
  await Promise.all(Array.from({ length: Math.min(4, queue.length) }, () => worker()))
}

let loadRevision = 0
async function load() {
  const revision = ++loadRevision
  if (!token.value) return
  loading.value = true; loadingMore.value = false; loadMoreError.value = ''; error.value = ''
  page.value = 1
  try {
    const params = new URLSearchParams({ page: '1', page_size: '24', channel: channel.value, project_id: project.value, status: status.value, q: query.value.trim() })
    const response = await fetch(`${base}/api/dashboard?${params}`, { headers: { Authorization: `Bearer ${token.value}` } })
    if (!response.ok) throw new Error(response.status === 401 ? 'Token 无效或已过期' : `请求失败（${response.status}）`)
    const data = await response.json()
    if (revision !== loadRevision) return
    tasks.value = data.tasks || []
    channels.value = data.channels || []
    projects.value = data.project_options || (data.projects || []).map((id: string, i: number) => ({ id, label: `未分类 · 历史项目 ${i + 1}（名称不可用）` }))
    counts.value = data.counts || {}
    total.value = data.total || 0
    connected.value = true
    void loadThumbnails(tasks.value)
    await nextTick()
    setupObserver()
  } catch (cause) {
    if (revision !== loadRevision) return
    connected.value = false
    error.value = cause instanceof Error ? cause.message : '连接失败'
  } finally {
    if (revision === loadRevision) loading.value = false
  }
}

async function loadMore() {
  if (!token.value || loading.value || loadingMore.value || !hasMore.value || document.hidden) return
  const revision = loadRevision
  const nextPage = page.value + 1
  loadingMore.value = true
  loadMoreError.value = ''
  try {
    const params = new URLSearchParams({ page: String(nextPage), page_size: '24', channel: channel.value, project_id: project.value, status: status.value, q: query.value.trim() })
    const response = await fetch(`${base}/api/dashboard?${params}`, { headers: { Authorization: `Bearer ${token.value}` } })
    if (!response.ok) throw new Error(`请求失败（${response.status}）`)
    const data = await response.json()
    if (revision !== loadRevision) return
    page.value = nextPage
    total.value = data.total || total.value
    tasks.value = mergeTasks(tasks.value, data.tasks || [])
    void loadThumbnails(data.tasks || [])
    await nextTick()
    if (scrollObserver && sentinelRef.value && hasMore.value) {
      scrollObserver.unobserve(sentinelRef.value)
      scrollObserver.observe(sentinelRef.value)
    }
  } catch (cause) {
    if (revision !== loadRevision) return
    loadMoreError.value = cause instanceof Error ? cause.message : '加载更多失败'
  } finally {
    if (revision === loadRevision) loadingMore.value = false
  }
}

function setupObserver() {
  if (typeof IntersectionObserver !== 'function' || !sentinelRef.value) return
  scrollObserver?.disconnect()
  scrollObserver = new IntersectionObserver(
    (entries) => {
      if (entries.some((e) => e.isIntersecting) && !loadMoreError.value) {
        void loadMore()
      }
    },
    { root: null, rootMargin: '1800px 0px', threshold: 0 },
  )
  scrollObserver.observe(sentinelRef.value)
}

function openTask(task: Task) { location.href = `${base}/dashboard/tasks/${encodeURIComponent(task.video_task_id)}` }
function saveToken() { localStorage.setItem(tokenKey, token.value); load() }
function clearToken() { token.value = ''; localStorage.removeItem(tokenKey); connected.value = false; tasks.value = []; revokeAllThumbnails() }
watch([channel, project, query, status], () => { tasks.value = []; load() })
onMounted(load)
onBeforeUnmount(() => { scrollObserver?.disconnect(); revokeAllThumbnails() })
</script>

<template>
  <header class="site-header"><div class="nav"><a class="brand" href="#"><span class="brand-mark" /><span>VerdantFlare</span><span class="brand-tag">VIDEO MCP</span></a><div class="nav-actions"><span class="pill-badge">{{ connected ? '已连接' : '等待连接' }}</span><button class="button" @click="token ? clearToken() : undefined">{{ token ? '清除 Token' : '未配置 Token' }}</button></div></div></header>
  <div class="workspace"><aside class="workspace-nav" aria-label="业务导航"><p>WORKSPACE</p><button v-for="item in [['tasks','01 / TASKS'],['models','02 / MODELS'],['mcp','03 / MCP']]" :key="item[0]" class="nav-link" :class="{ selected: active === item[0] }" @click="active = item[0] as typeof active">{{ item[1] }}</button></aside>
    <main class="market-page">
      <section v-if="!token" class="market-wrap business-section"><h1>连接 Video MCP</h1><p class="business-muted">使用工作区 Token 读取真实任务、模型和服务状态。</p><form class="token-form" @submit.prevent="saveToken"><input data-testid="token-input" v-model="token" type="password" autocomplete="off" placeholder="Video MCP Token" required /><button data-testid="connect-token" class="button primary">连接</button></form></section>
      <section v-else-if="active === 'tasks'" data-testid="tasks-view" class="market-wrap business-section"><h1>任务</h1><p class="business-muted">视频生成与处理任务 · {{ total }} 条</p><div class="market-toolbar"><label class="market-search"><span>任务 / 项目 / Prompt</span><input data-testid="task-search" v-model="query" type="search" placeholder="搜索任务编号、提示词或项目…" /></label><label><span>渠道</span><select data-testid="channel-filter" v-model="channel"><option value="all">全部渠道</option><option v-for="item in channels" :key="item.id" :value="item.id">{{ item.label }}</option></select></label><label><span>所属项目</span><select data-testid="project-filter" v-model="project"><option value="all">全部项目</option><option v-for="item in projects" :key="item.id" :value="item.id">{{ item.label }}</option></select></label></div><div class="market-filter-row"><button v-for="item in [['all','全部状态'],['queued','⏳ 排队中'],['running','⚡ 渲染中'],['succeeded','✓ 已完成'],['failed','✕ 失败'],['cancelled','已取消']]" :key="item[0]" :class="{ active: status === item[0] }" @click="status = item[0]">{{ item[1] }} <span v-if="item[0] !== 'all'">{{ counts[item[0]] || 0 }}</span></button></div><p v-if="loading" class="empty">正在读取任务…</p><p v-else-if="error" class="empty">{{ error }} <button class="button" @click="load">重试</button></p><p v-else-if="!filtered.length" class="empty">暂无符合条件的任务。</p><div v-else class="model-results grid"><article v-for="task in filtered" :key="task.video_task_id" data-testid="task-card" class="model-card" role="link" tabindex="0" :aria-label="`查看 ${task.title || '视频任务'}`" @click="openTask(task)" @keydown.enter="openTask(task)"><div class="card-thumb-wrap"><img v-if="thumbUrls[task.video_task_id]" :src="thumbUrls[task.video_task_id]" alt="生成视频首帧" class="card-thumb-img" /><span v-else class="thumb-placeholder" :class="{ 'thumb-error': thumbErrors[task.video_task_id] }"><span>{{ thumbStatusText(task) }}</span><button v-if="thumbErrors[task.video_task_id]" type="button" class="button thumb-retry-btn" @click.stop="fetchThumbnail(task, true)">重试首帧</button></span><div v-if="generation.has(task.service)" class="task-identity generation-identity" :title="`${modelName(task)} × ${routeName(task)}`"><strong class="identity-model">{{ modelName(task) }}</strong><span class="identity-separator" aria-hidden="true">×</span><strong class="identity-channel"><i v-if="task.presentation?.channel_icon === 'higgsfield'" class="channel-logo" aria-hidden="true" v-html="higgsfieldLogo" /><svg v-else-if="task.presentation?.channel_icon === 'local'" class="channel-logo" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8m-4-4v4"/></svg><svg v-else-if="isFalChannel(task)" class="channel-logo" width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><rect width="24" height="24" rx="6" fill="#6C3AED"/><path d="M8.5 7H15.5M8.5 12H13.5M8.5 7V17" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/><circle cx="15.5" cy="15.5" r="1.8" fill="#4CE39A"/></svg>{{ routeName(task) }}</strong></div><div v-else class="task-identity"><div v-for="line in identity(task)" v-show="line" :key="line"><span>{{ line.split('：')[0] }}：</span><strong>{{ line.split('：').slice(1).join('：') }}</strong></div></div><span class="card-badges"><span class="badge-status" :class="task.status">{{ labels[task.status] || task.status }}</span></span><div class="card-times"><span>{{ date(task.created_at) }}</span><span>总耗时 {{ duration(task) }}</span></div></div><div class="card-body"><div><div class="card-title-row">{{ task.title || '视频任务' }}</div><p class="card-prompt">{{ generation.has(task.service) ? (task.prompt || '未提供提示词') : operations[task.service] || '视频处理' }}</p></div><div class="card-footer"><div class="model-tags"><span v-if="task.presentation?.generation_code" class="generation-type" tabindex="0" :title="task.presentation.generation_label" :data-tooltip="task.presentation.generation_label" :aria-label="`${task.presentation.generation_code} · ${task.presentation.generation_label}`">{{ task.presentation.generation_code }}</span><span v-if="generation.has(task.service) && taskResolution(task)">{{ taskResolution(task) }}</span><span v-if="generation.has(task.service)">{{ task.aspect_ratio || '—' }}</span><span v-if="task.duration_seconds">{{ task.duration_seconds }}S</span><span v-if="generation.has(task.service)">{{ taskFps(task) }}</span></div><span class="card-footer-action">查看</span></div></div></article></div><div ref="sentinelRef" class="market-filter-row"><span v-if="loadingMore">正在加载更多任务…（已加载 {{ filtered.length }} / 共 {{ total }} 条）</span><template v-else-if="loadMoreError"><span>加载更多失败：{{ loadMoreError }}（已保留 {{ filtered.length }} / 共 {{ total }} 条）</span><button class="button" @click="loadMore">重试加载</button></template><template v-else-if="hasMore"><span>已加载 {{ filtered.length }} / 共 {{ total }} 条</span><button class="button" @click="loadMore">加载更多</button></template><span v-else-if="total > 0">已加载全部 {{ total }} 条任务</span></div></section>
      <section v-else-if="active === 'models'" data-testid="models-view" class="market-wrap business-section"><h1>模型</h1><p class="business-muted">按模型或渠道查看用途、规格与资料状态。</p><ModelCatalog :token="token" :base="base" /></section>
      <section v-else data-testid="mcp-view" class="market-wrap business-section"><h1>MCP 服务</h1><p class="business-muted">连接状态：{{ connected ? '已连接' : '未知' }}</p><div class="business-card"><h3>Video MCP Server</h3><p>任务接入、编排、Artifact 和服务观测</p></div></section>
    </main>
  </div>
</template>

<style scoped>
.token-form { display:flex; gap:12px; max-width:560px; margin-top:24px }
.token-form input { flex:1; min-width:0; border:1px solid var(--line); border-radius:8px; padding:10px 12px; background:var(--subtle); color:var(--ink) }
.task-identity { pointer-events:auto }
.card-badges { position:absolute; right:12px; top:12px }
.badge-status { display:inline-flex; padding:4px 8px; border-radius:999px; background:var(--subtle); color:var(--muted); font-size:11px }
.badge-status.succeeded { color:var(--accent) }
.badge-status.failed { color:var(--danger) }
.empty { padding:42px 20px; color:var(--muted); text-align:center }
@media (max-width: 720px) { .token-form { flex-direction:column } }
</style>


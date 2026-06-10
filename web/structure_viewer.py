"""
Streamlit NGL 蛋白对接结构可视化组件。
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

from streamlit.components.v1 import html

from docking.interface import InterfaceResidue, InterfaceResult


NGL_CDN = "https://cdn.jsdelivr.net/npm/ngl@2.3.1/dist/ngl.js"

VIEWER_LABELS = {
    "zh": {
        "render": "表示",
        "cartoon": "卡通",
        "hybrid": "卡通 + 表面",
        "surface": "表面",
        "focus_interface": "聚焦界面",
        "global_view": "全局视图",
        "auto_spin": "自动旋转",
        "export_format": "图片格式",
        "export_image": "保存高清图",
        "export_png": "PNG",
        "export_jpg": "JPG",
        "exporting_image": "正在保存...",
        "export_image_failed": "图片保存失败：",
        "interface_sticks": "界面 sticks",
        "contact_lines": "接触连线",
        "cofactors": "辅因子",
        "viewer_hint": "拖拽旋转 · 滚轮缩放 · 右键平移",
        "description": "基于完整对接复合物渲染；受体与配体分色显示，界面残基与接触对可单独开关。",
        "interface_stats": "界面统计",
        "interface_residues_total": "界面残基总数",
        "contact_pairs_total": "接触对数量",
        "receptor_interface_residues": "受体侧界面残基",
        "ligand_interface_residues": "配体侧界面残基",
        "chain_roles": "链角色",
        "receptor": "受体",
        "ligand": "配体",
        "shortest_contacts": "最短接触对",
        "distance": "距离",
        "type": "类型",
        "none": "无",
        "docking_structure": "对接结构",
        "no_contacts": "未识别到可展示的 CA 接触对。",
        "structure_load_failed": "结构加载失败：",
        "contact_hbond": "氢键",
        "contact_hydrophobic": "疏水",
        "contact_electrostatic": "静电",
        "contact_van_der_waals": "范德华",
        "contact_contact": "接触",
    },
    "en": {
        "render": "Representation",
        "cartoon": "Cartoon",
        "hybrid": "Cartoon + Surface",
        "surface": "Surface",
        "focus_interface": "Focus Interface",
        "global_view": "Global View",
        "auto_spin": "Auto Spin",
        "export_format": "Image Format",
        "export_image": "Save HD Image",
        "export_png": "PNG",
        "export_jpg": "JPG",
        "exporting_image": "Saving...",
        "export_image_failed": "Failed to save image: ",
        "interface_sticks": "Interface sticks",
        "contact_lines": "Contact lines",
        "cofactors": "Cofactors",
        "viewer_hint": "Drag to rotate · Scroll to zoom · Right-drag to pan",
        "description": "Rendered from the complete docked complex; receptor, ligand, interface residues, and contacts are shown separately.",
        "interface_stats": "Interface Statistics",
        "interface_residues_total": "Total interface residues",
        "contact_pairs_total": "Contact pairs",
        "receptor_interface_residues": "Receptor interface residues",
        "ligand_interface_residues": "Ligand interface residues",
        "chain_roles": "Chain Roles",
        "receptor": "Receptor",
        "ligand": "Ligand",
        "shortest_contacts": "Shortest Contacts",
        "distance": "Distance",
        "type": "Type",
        "none": "None",
        "docking_structure": "Docked Structure",
        "no_contacts": "No displayable CA contact pairs were identified.",
        "structure_load_failed": "Failed to load structure: ",
        "contact_hbond": "H-bond",
        "contact_hydrophobic": "Hydrophobic",
        "contact_electrostatic": "Electrostatic",
        "contact_van_der_waals": "Van der Waals",
        "contact_contact": "Contact",
    },
}


def _parse_pdb_atoms(pdb_content: str) -> List[Dict[str, object]]:
    atoms: List[Dict[str, object]] = []
    for line in pdb_content.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            atoms.append(
                {
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": (line[21:22].strip() or "A"),
                    "resseq": int(line[22:26].strip()),
                    "x": float(line[30:38].strip()),
                    "y": float(line[38:46].strip()),
                    "z": float(line[46:54].strip()),
                }
            )
        except ValueError:
            continue
    return atoms


def _extract_ca_coordinates(pdb_content: str) -> Dict[str, Dict[str, float]]:
    coords: Dict[str, Dict[str, float]] = {}
    for atom in _parse_pdb_atoms(pdb_content):
        if atom["name"] != "CA":
            continue
        label = f"{atom['chain']}:{atom['resname']}{atom['resseq']}"
        coords[label] = {
            "x": float(atom["x"]),
            "y": float(atom["y"]),
            "z": float(atom["z"]),
        }
    return coords


def _chain_selection(chains: Iterable[str]) -> str:
    unique = [chain for chain in sorted(set(chains)) if chain]
    return " or ".join(f":{chain}" for chain in unique)


def _interface_selection(residues: Iterable[InterfaceResidue]) -> str:
    seen = set()
    parts: List[str] = []
    for residue in sorted(residues, key=lambda item: (item.chain_id, item.resseq, item.resname)):
        key = (residue.chain_id, residue.resseq, residue.resname)
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"{residue.resseq}:{residue.chain_id}")
    return " or ".join(parts)


def _contact_type_lookup(interface_result: InterfaceResult) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for residue in interface_result.receptor_interface:
        lookup[f"{residue.chain_id}:{residue.resname}{residue.resseq}"] = residue.contact_type
    return lookup


def _build_contact_rows(
    interface_result: InterfaceResult,
    pdb_content: str,
    limit: int = 40,
) -> List[Dict[str, object]]:
    ca_coords = _extract_ca_coordinates(pdb_content)
    type_lookup = _contact_type_lookup(interface_result)
    rows: List[Dict[str, object]] = []

    for rec_label, lig_label, distance in sorted(interface_result.contact_pairs, key=lambda item: item[2])[:limit]:
        if rec_label not in ca_coords or lig_label not in ca_coords:
            continue
        rows.append(
            {
                "receptor_label": rec_label,
                "ligand_label": lig_label,
                "distance": round(float(distance), 2),
                "contact_type": type_lookup.get(rec_label, "contact"),
                "receptor_coord": ca_coords[rec_label],
                "ligand_coord": ca_coords[lig_label],
            }
        )
    return rows


def build_viewer_payload(
    pdb_content: str,
    interface_result: InterfaceResult,
    receptor_chains: Iterable[str],
    ligand_chains: Iterable[str],
    pose_title: str,
    metrics: Optional[List[Dict[str, str]]] = None,
    language: str = "zh",
) -> Dict[str, object]:
    language = language if language in VIEWER_LABELS else "zh"
    receptor_chain_list = sorted(set(receptor_chains))
    ligand_chain_list = sorted(set(ligand_chains))
    receptor_interface_sel = _interface_selection(interface_result.receptor_interface)
    ligand_interface_sel = _interface_selection(interface_result.ligand_interface)
    combined_sel = " or ".join(part for part in [receptor_interface_sel, ligand_interface_sel] if part)

    return {
        "pdbText": pdb_content,
        "poseTitle": pose_title,
        "metrics": metrics or [],
        "language": language,
        "labels": VIEWER_LABELS[language],
        "receptorChains": receptor_chain_list,
        "ligandChains": ligand_chain_list,
        "receptorChainSelection": _chain_selection(receptor_chain_list),
        "ligandChainSelection": _chain_selection(ligand_chain_list),
        "interface": {
            "receptorSelection": receptor_interface_sel,
            "ligandSelection": ligand_interface_sel,
            "combinedSelection": combined_sel,
            "contactPairsTotal": len(interface_result.contact_pairs),
            "interfaceResiduesTotal": interface_result.n_interface_residues,
            "receptorResidues": len(interface_result.receptor_interface),
            "ligandResidues": len(interface_result.ligand_interface),
        },
        "contacts": _build_contact_rows(interface_result, pdb_content),
    }


def build_structure_viewer_html(payload: Dict[str, object], height_px: int = 760) -> str:
    normalized_payload = dict(payload)
    language = normalized_payload.get("language", "zh")
    language = language if language in VIEWER_LABELS else "zh"
    normalized_payload["language"] = language
    normalized_payload["labels"] = {
        **VIEWER_LABELS[language],
        **normalized_payload.get("labels", {}),
    }
    payload_json = json.dumps(normalized_payload, ensure_ascii=False)
    template = """
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <script src="__NGL_CDN__"></script>
    <style>
      :root {
        color-scheme: dark;
        --bg-app: #08101b;
        --bg-panel: rgba(12, 20, 33, 0.88);
        --bg-panel-strong: #101926;
        --bg-elevated: rgba(18, 29, 46, 0.96);
        --border: rgba(126, 156, 205, 0.16);
        --border-strong: rgba(126, 156, 205, 0.26);
        --text: #e8edf6;
        --text-muted: #8ea0bd;
        --receptor: #67a8ff;
        --ligand: #ffb45e;
        --contact: #ffd166;
        --success: #52d3a6;
        --hbond: #6ed6ff;
        --hydrophobic: #ffd166;
        --electrostatic: #ff8f70;
        --van-der-waals: #caa7ff;
      }

      * {
        box-sizing: border-box;
      }

      html,
      body {
        margin: 0;
        width: 100%;
        height: 100%;
        background: var(--bg-app);
        color: var(--text);
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }

      .viewer-shell {
        width: 100%;
        height: __HEIGHT_PX__px;
        min-height: 0;
        display: grid;
        grid-template-columns: minmax(0, 1fr) 360px;
        overflow: hidden;
        background: var(--bg-app);
        border: 1px solid var(--border);
        border-radius: 8px;
      }

      @media (max-width: 720px) {
        .viewer-shell {
          grid-template-columns: 1fr;
          grid-template-rows: minmax(560px, 1fr) auto;
          height: auto;
        }
      }

      .viewer-pane {
        position: relative;
        min-height: 0;
        border-right: 1px solid var(--border);
      }

      @media (max-width: 720px) {
        .viewer-pane {
          min-height: 560px;
          border-right: none;
          border-bottom: 1px solid var(--border);
        }
      }

      .toolbar {
        position: absolute;
        top: 12px;
        left: 12px;
        right: 12px;
        z-index: 20;
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        padding: 10px 12px;
        background: rgba(10, 16, 28, 0.72);
        border: 1px solid var(--border);
        border-radius: 8px;
        backdrop-filter: blur(12px);
      }

      .toolbar-group {
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .toolbar label {
        font-size: 11px;
        color: var(--text-muted);
      }

      .toolbar button,
      .toolbar select {
        appearance: none;
        border: 1px solid var(--border);
        background: var(--bg-panel-strong);
        color: var(--text);
        border-radius: 8px;
        padding: 7px 10px;
        font-size: 12px;
        line-height: 1.15;
        min-height: 30px;
      }

      .toolbar button {
        cursor: pointer;
      }

      .toolbar button.is-active {
        border-color: rgba(103, 168, 255, 0.55);
        color: var(--receptor);
        background: rgba(103, 168, 255, 0.08);
      }

      .toolbar button:disabled {
        cursor: progress;
        opacity: 0.62;
      }

      .toolbar .export-group {
        margin-left: auto;
      }

      #btn-export-image {
        min-width: 98px;
      }

      .toolbar .toggle {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0;
        border: none;
        background: transparent;
      }

      .toolbar .toggle input {
        width: 14px;
        height: 14px;
        margin: 0;
        accent-color: var(--receptor);
      }

      #viewport {
        width: 100%;
        height: 100%;
        min-height: 560px;
      }

      .viewer-hint {
        position: absolute;
        right: 12px;
        bottom: 12px;
        z-index: 10;
        padding: 8px 10px;
        font-size: 11px;
        color: var(--text-muted);
        background: rgba(9, 15, 26, 0.74);
        border: 1px solid var(--border);
        border-radius: 8px;
      }

      .analysis-pane {
        min-width: 0;
        min-height: 0;
        height: 100%;
        display: flex;
        flex-direction: column;
        gap: 12px;
        padding: 14px;
        overflow-x: hidden;
        overflow-y: auto;
        background: var(--bg-panel);
      }

      .title-block h2 {
        margin: 0;
        font-size: 16px;
        font-weight: 600;
      }

      .title-block p {
        margin: 6px 0 0;
        font-size: 12px;
        line-height: 1.5;
        color: var(--text-muted);
      }

      .metric-strip {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
      }

      .metric {
        padding: 10px;
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: 8px;
      }

      .metric-label {
        font-size: 11px;
        color: var(--text-muted);
        margin-bottom: 5px;
      }

      .metric-value {
        font-size: 18px;
        font-weight: 600;
      }

      .panel {
        min-width: 0;
        flex: 0 0 auto;
        overflow: hidden;
        border: 1px solid var(--border);
        background: rgba(16, 25, 38, 0.92);
        border-radius: 8px;
      }

      .panel:has(.contacts-table-wrap) {
        min-height: 0;
        flex: 1 1 0;
        display: flex;
        flex-direction: column;
      }

      .panel:has(.contacts-table-wrap) .panel-head {
        flex: 0 0 auto;
      }

      .panel-head {
        padding: 10px 12px;
        border-bottom: 1px solid var(--border);
        font-size: 12px;
        font-weight: 600;
      }

      .panel-body {
        padding: 10px 12px;
      }

      .stats-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
      }

      .stat-row {
        padding: 8px;
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.015);
        border-radius: 6px;
      }

      .stat-row strong {
        display: block;
        font-size: 15px;
      }

      .stat-row span {
        display: block;
        margin-top: 3px;
        font-size: 11px;
        color: var(--text-muted);
      }

      .chain-list {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }

      .chain-badge {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding: 8px 10px;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--border);
        border-radius: 6px;
        font-size: 12px;
      }

      .chain-badge .label {
        display: inline-flex;
        align-items: center;
        gap: 8px;
      }

      .swatch {
        width: 10px;
        height: 10px;
        display: inline-block;
      }

      .contacts-table-wrap {
        position: relative;
        isolation: isolate;
        padding-top: 0;
        min-height: 0;
        max-height: none;
        flex: 1 1 auto;
        overflow-x: hidden;
        overflow-y: auto;
        overscroll-behavior: contain;
        scrollbar-gutter: stable;
      }

      table {
        width: 100%;
        table-layout: fixed;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 12px;
      }

      th,
      td {
        text-align: left;
        padding: 8px 0;
        border-bottom: 1px solid rgba(126, 156, 205, 0.09);
        vertical-align: top;
        overflow-wrap: anywhere;
      }

      thead {
        position: sticky;
        top: 0;
        z-index: 10;
        background: var(--bg-panel-strong);
      }

      thead th {
        position: relative;
        z-index: 11;
        background: var(--bg-panel-strong);
        box-shadow: 0 1px 0 var(--border-strong);
        color: var(--text-muted);
        font-size: 11px;
        font-weight: 500;
      }

      tbody {
        position: relative;
        z-index: 0;
      }

      .contact-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 72px;
        padding: 3px 8px;
        border: 1px solid var(--border-strong);
        border-radius: 6px;
        font-size: 10px;
        text-transform: uppercase;
      }

      .contact-badge.hbond { color: var(--hbond); }
      .contact-badge.hydrophobic { color: var(--hydrophobic); }
      .contact-badge.electrostatic { color: var(--electrostatic); }
      .contact-badge.van_der_waals,
      .contact-badge.contact { color: var(--van-der-waals); }

      .empty-line {
        font-size: 12px;
        color: var(--text-muted);
      }

      @media (max-width: 720px) {
        .viewer-shell {
          overflow: visible;
        }

        .toolbar {
          position: relative;
          top: auto;
          left: auto;
          right: auto;
          margin: 10px;
        }

        #viewport {
          height: 520px;
          min-height: 520px;
        }

        .analysis-pane {
          height: auto;
          overflow: visible;
        }

        .panel:has(.contacts-table-wrap) {
          max-height: 340px;
        }
      }
    </style>
  </head>
  <body>
    <div class="viewer-shell">
      <section class="viewer-pane">
        <div class="toolbar">
          <div class="toolbar-group">
            <label for="render-mode" id="render-mode-label"></label>
            <select id="render-mode">
              <option value="cartoon" id="render-cartoon"></option>
              <option value="hybrid" id="render-hybrid"></option>
              <option value="surface" id="render-surface"></option>
            </select>
          </div>
          <div class="toolbar-group">
            <button type="button" id="btn-focus-interface"></button>
            <button type="button" id="btn-reset-view"></button>
            <button type="button" id="btn-spin"></button>
          </div>
          <div class="toolbar-group">
            <label class="toggle"><input type="checkbox" id="toggle-interface" checked /><span id="toggle-interface-label"></span></label>
            <label class="toggle"><input type="checkbox" id="toggle-contacts" checked /><span id="toggle-contacts-label"></span></label>
            <label class="toggle"><input type="checkbox" id="toggle-hetero" /><span id="toggle-hetero-label"></span></label>
          </div>
          <div class="toolbar-group export-group">
            <label for="export-format" id="export-format-label"></label>
            <select id="export-format">
              <option value="png" id="export-png"></option>
              <option value="jpg" id="export-jpg"></option>
            </select>
            <button type="button" id="btn-export-image"></button>
          </div>
        </div>
        <div id="viewport"></div>
        <div class="viewer-hint" id="viewer-hint"></div>
      </section>

      <aside class="analysis-pane">
        <div class="title-block">
          <h2 id="pose-title"></h2>
          <p id="viewer-description"></p>
        </div>

        <div class="metric-strip" id="metric-strip"></div>

        <section class="panel">
          <div class="panel-head" id="interface-stats-title"></div>
          <div class="panel-body">
            <div class="stats-grid">
              <div class="stat-row">
                <strong id="interface-total"></strong>
                <span id="interface-total-label"></span>
              </div>
              <div class="stat-row">
                <strong id="contact-total"></strong>
                <span id="contact-total-label"></span>
              </div>
              <div class="stat-row">
                <strong id="receptor-total"></strong>
                <span id="receptor-total-label"></span>
              </div>
              <div class="stat-row">
                <strong id="ligand-total"></strong>
                <span id="ligand-total-label"></span>
              </div>
            </div>
          </div>
        </section>

        <section class="panel">
          <div class="panel-head" id="chain-roles-title"></div>
          <div class="panel-body chain-list" id="chain-list"></div>
        </section>

        <section class="panel">
          <div class="panel-head" id="shortest-contacts-title"></div>
          <div class="panel-body contacts-table-wrap">
            <table>
              <thead>
                <tr>
                  <th id="contact-receptor-heading"></th>
                  <th id="contact-ligand-heading"></th>
                  <th id="contact-distance-heading"></th>
                  <th id="contact-type-heading"></th>
                </tr>
              </thead>
              <tbody id="contact-table-body"></tbody>
            </table>
          </div>
        </section>
      </aside>
    </div>

    <script>
      const payload = __PAYLOAD_JSON__;
      const labels = payload.labels || {};
      const receptorColor = "#67a8ff";
      const ligandColor = "#ffb45e";
      const contactColor = [1.0, 0.82, 0.4];

      let stage = null;
      let component = null;
      let shapeComponent = null;
      let spinning = false;
      const reps = {};

      function escapeHtml(text) {
        return String(text)
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;");
      }

      function renderMetrics() {
        const strip = document.getElementById("metric-strip");
        const metrics = Array.isArray(payload.metrics) ? payload.metrics : [];
        strip.innerHTML = "";
        metrics.forEach(function(metric) {
          const el = document.createElement("div");
          el.className = "metric";
          el.innerHTML =
            '<div class="metric-label">' + escapeHtml(metric.label) + "</div>" +
            '<div class="metric-value">' + escapeHtml(metric.value) + "</div>";
          strip.appendChild(el);
        });
      }

      function applyLabels() {
        document.documentElement.lang = payload.language === "en" ? "en" : "zh-CN";
        const textById = {
          "render-mode-label": labels.render,
          "render-cartoon": labels.cartoon,
          "render-hybrid": labels.hybrid,
          "render-surface": labels.surface,
          "btn-focus-interface": labels.focus_interface,
          "btn-reset-view": labels.global_view,
          "btn-spin": labels.auto_spin,
          "export-format-label": labels.export_format,
          "export-png": labels.export_png,
          "export-jpg": labels.export_jpg,
          "btn-export-image": labels.export_image,
          "toggle-interface-label": labels.interface_sticks,
          "toggle-contacts-label": labels.contact_lines,
          "toggle-hetero-label": labels.cofactors,
          "viewer-hint": labels.viewer_hint,
          "viewer-description": labels.description,
          "interface-stats-title": labels.interface_stats,
          "interface-total-label": labels.interface_residues_total,
          "contact-total-label": labels.contact_pairs_total,
          "receptor-total-label": labels.receptor_interface_residues,
          "ligand-total-label": labels.ligand_interface_residues,
          "chain-roles-title": labels.chain_roles,
          "shortest-contacts-title": labels.shortest_contacts,
          "contact-receptor-heading": labels.receptor,
          "contact-ligand-heading": labels.ligand,
          "contact-distance-heading": labels.distance,
          "contact-type-heading": labels.type
        };
        Object.entries(textById).forEach(function(entry) {
          const element = document.getElementById(entry[0]);
          if (element) element.textContent = entry[1] || "";
        });
      }

      function renderSummary() {
        applyLabels();
        document.getElementById("pose-title").textContent = payload.poseTitle || labels.docking_structure;
        document.getElementById("interface-total").textContent = String(payload.interface.interfaceResiduesTotal || 0);
        document.getElementById("contact-total").textContent = String(payload.interface.contactPairsTotal || 0);
        document.getElementById("receptor-total").textContent = String(payload.interface.receptorResidues || 0);
        document.getElementById("ligand-total").textContent = String(payload.interface.ligandResidues || 0);
        renderMetrics();

        const chainList = document.getElementById("chain-list");
        chainList.innerHTML = "";
        [
          { role: labels.receptor, color: receptorColor, chains: payload.receptorChains || [] },
          { role: labels.ligand, color: ligandColor, chains: payload.ligandChains || [] }
        ].forEach(function(group) {
          const el = document.createElement("div");
          el.className = "chain-badge";
          el.innerHTML =
            '<div class="label"><span class="swatch" style="background:' + group.color + '"></span>' +
            escapeHtml(group.role) + "</div>" +
            '<div>' + escapeHtml(group.chains.join(", ") || labels.none) + "</div>";
          chainList.appendChild(el);
        });

        const tbody = document.getElementById("contact-table-body");
        tbody.innerHTML = "";
        const contacts = payload.contacts || [];
        if (!contacts.length) {
          const tr = document.createElement("tr");
          tr.innerHTML = '<td colspan="4" class="empty-line">' + escapeHtml(labels.no_contacts) + "</td>";
          tbody.appendChild(tr);
          return;
        }
        contacts.slice(0, 18).forEach(function(contact) {
          const tr = document.createElement("tr");
          const contactType = labels["contact_" + contact.contact_type] ||
            contact.contact_type.replaceAll("_", " ");
          tr.innerHTML =
            "<td>" + escapeHtml(contact.receptor_label) + "</td>" +
            "<td>" + escapeHtml(contact.ligand_label) + "</td>" +
            "<td>" + escapeHtml(String(contact.distance)) + " Å</td>" +
            '<td><span class="contact-badge ' + escapeHtml(contact.contact_type) + '">' +
            escapeHtml(contactType) + "</span></td>";
          tbody.appendChild(tr);
        });
      }

      function initStage() {
        if (stage) return;
        stage = new NGL.Stage("viewport");
        stage.setParameters({
          backgroundColor: "#08101b",
          quality: "high",
          sampleLevel: 1
        });
        window.addEventListener("resize", function() {
          stage.handleResize();
        });
      }

      function setVisibility(rep, visible) {
        if (rep && typeof rep.setVisibility === "function") {
          rep.setVisibility(Boolean(visible));
        }
      }

      function buildContactShape() {
        if (!stage || !component || !payload.contacts || !payload.contacts.length) {
          return;
        }
        const shape = new NGL.Shape("contacts");
        payload.contacts.slice(0, 32).forEach(function(contact) {
          const a = contact.receptor_coord;
          const b = contact.ligand_coord;
          shape.addSphere([a.x, a.y, a.z], [0.41, 0.66, 1.0], 0.42);
          shape.addSphere([b.x, b.y, b.z], [1.0, 0.71, 0.37], 0.42);
          shape.addCylinder([a.x, a.y, a.z], [b.x, b.y, b.z], contactColor, 0.08);
        });
        shapeComponent = stage.addComponentFromObject(shape);
        reps.contactShape = shapeComponent.addRepresentation("buffer");
      }

      function applyRenderMode(mode) {
        const cartoonVisible = mode === "cartoon" || mode === "hybrid";
        const surfaceVisible = mode === "surface" || mode === "hybrid";
        setVisibility(reps.receptorCartoon, cartoonVisible);
        setVisibility(reps.ligandCartoon, cartoonVisible);
        setVisibility(reps.receptorSurface, surfaceVisible);
        setVisibility(reps.ligandSurface, surfaceVisible);
      }

      function toggleSpin() {
        spinning = !spinning;
        const btn = document.getElementById("btn-spin");
        btn.classList.toggle("is-active", spinning);
        if (stage) {
          stage.setSpin(spinning ? [0.002, 0.001, 0] : false);
        }
      }

      function imageFileBase() {
        const raw = payload.poseTitle || labels.docking_structure || "docked_structure";
        const safe = String(raw)
          .toLowerCase()
          .replace(/[^a-z0-9]+/gi, "_")
          .replace(/^_+|_+$/g, "");
        return safe || "docked_structure";
      }

      function downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.setTimeout(function() {
          URL.revokeObjectURL(url);
        }, 1000);
      }

      function blobFromImageResult(result) {
        if (result instanceof Blob) {
          return Promise.resolve(result);
        }
        if (typeof result === "string" && result.startsWith("data:")) {
          return fetch(result).then(function(response) {
            return response.blob();
          });
        }
        return Promise.reject(new Error("Unsupported image result"));
      }

      function convertPngToJpeg(pngBlob) {
        return new Promise(function(resolve, reject) {
          const url = URL.createObjectURL(pngBlob);
          const image = new Image();
          image.onload = function() {
            const canvas = document.createElement("canvas");
            canvas.width = image.naturalWidth || image.width;
            canvas.height = image.naturalHeight || image.height;
            const ctx = canvas.getContext("2d");
            ctx.fillStyle = "#08101b";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(image, 0, 0);
            canvas.toBlob(function(jpegBlob) {
              URL.revokeObjectURL(url);
              if (jpegBlob) resolve(jpegBlob);
              else reject(new Error("JPEG conversion returned an empty image"));
            }, "image/jpeg", 0.95);
          };
          image.onerror = function() {
            URL.revokeObjectURL(url);
            reject(new Error("Could not read the rendered image"));
          };
          image.src = url;
        });
      }

      function setExportBusy(isBusy) {
        const button = document.getElementById("btn-export-image");
        button.disabled = Boolean(isBusy);
        button.textContent = isBusy ? (labels.exporting_image || "Saving...") : (labels.export_image || "Save HD Image");
      }

      async function exportHighResolutionImage() {
        if (!stage || !component || typeof stage.makeImage !== "function") {
          return;
        }
        const format = document.getElementById("export-format").value === "jpg" ? "jpg" : "png";
        const wasSpinning = spinning;
        try {
          setExportBusy(true);
          if (wasSpinning) {
            stage.setSpin(false);
          }
          stage.handleResize();
          await new Promise(function(resolve) {
            requestAnimationFrame(function() {
              requestAnimationFrame(resolve);
            });
          });
          const result = await stage.makeImage({
            factor: 4,
            antialias: true,
            trim: false,
            transparent: false
          });
          const pngBlob = await blobFromImageResult(result);
          const blob = format === "jpg" ? await convertPngToJpeg(pngBlob) : pngBlob;
          downloadBlob(blob, imageFileBase() + "_4x." + format);
        } catch (error) {
          console.error(error);
          window.alert((labels.export_image_failed || "Failed to save image: ") + (error && error.message ? error.message : String(error)));
        } finally {
          if (wasSpinning && stage) {
            stage.setSpin([0.002, 0.001, 0]);
          }
          setExportBusy(false);
        }
      }

      function wireControls() {
        document.getElementById("render-mode").addEventListener("change", function(event) {
          applyRenderMode(event.target.value);
        });
        document.getElementById("btn-spin").addEventListener("click", toggleSpin);
        document.getElementById("btn-reset-view").addEventListener("click", function() {
          if (component) component.autoView();
        });
        document.getElementById("btn-focus-interface").addEventListener("click", function() {
          if (!component) return;
          const selection = payload.interface.combinedSelection;
          if (selection) component.autoView(selection, 400);
          else component.autoView();
        });
        document.getElementById("toggle-interface").addEventListener("change", function(event) {
          setVisibility(reps.receptorInterface, event.target.checked);
          setVisibility(reps.ligandInterface, event.target.checked);
        });
        document.getElementById("toggle-contacts").addEventListener("change", function(event) {
          setVisibility(reps.contactShape, event.target.checked);
        });
        document.getElementById("toggle-hetero").addEventListener("change", function(event) {
          setVisibility(reps.hetero, event.target.checked);
        });
        document.getElementById("btn-export-image").addEventListener("click", exportHighResolutionImage);
      }

      function loadStructure() {
        initStage();
        renderSummary();
        wireControls();

        const pdbBlob = new Blob([payload.pdbText], { type: "text/plain" });
        stage.removeAllComponents();
        stage.loadFile(pdbBlob, { ext: "pdb" }).then(function(comp) {
          component = comp;
          const receptorSele = payload.receptorChainSelection || "polymer";
          const ligandSele = payload.ligandChainSelection || "";
          const receptorInterfaceSele = payload.interface.receptorSelection || "";
          const ligandInterfaceSele = payload.interface.ligandSelection || "";

          reps.receptorCartoon = component.addRepresentation("cartoon", {
            sele: receptorSele,
            colorScheme: "uniform",
            colorValue: receptorColor,
            quality: "high",
            opacity: 1.0
          });
          if (ligandSele) {
            reps.ligandCartoon = component.addRepresentation("cartoon", {
              sele: ligandSele,
              colorScheme: "uniform",
              colorValue: ligandColor,
              quality: "high",
              opacity: 1.0
            });
          }
          reps.receptorSurface = component.addRepresentation("surface", {
            sele: receptorSele,
            colorScheme: "uniform",
            colorValue: receptorColor,
            opacity: 0.18,
            visible: false,
            roughness: 1.0,
            quality: "medium"
          });
          if (ligandSele) {
            reps.ligandSurface = component.addRepresentation("surface", {
              sele: ligandSele,
              colorScheme: "uniform",
              colorValue: ligandColor,
              opacity: 0.24,
              visible: false,
              roughness: 1.0,
              quality: "medium"
            });
          }
          if (receptorInterfaceSele) {
            reps.receptorInterface = component.addRepresentation("licorice", {
              sele: receptorInterfaceSele,
              colorScheme: "uniform",
              colorValue: receptorColor,
              multipleBond: true,
              scale: 0.28
            });
          }
          if (ligandInterfaceSele) {
            reps.ligandInterface = component.addRepresentation("licorice", {
              sele: ligandInterfaceSele,
              colorScheme: "uniform",
              colorValue: ligandColor,
              multipleBond: true,
              scale: 0.28
            });
          }
          reps.hetero = component.addRepresentation("ball+stick", {
            sele: "hetero and not (water or ion)",
            visible: false,
            opacity: 0.92,
            scale: 0.26
          });

          buildContactShape();
          applyRenderMode(document.getElementById("render-mode").value);
          component.autoView(payload.interface.combinedSelection || undefined);
          requestAnimationFrame(function() {
            stage.handleResize();
            component.autoView(payload.interface.combinedSelection || undefined);
          });
        }).catch(function(error) {
          console.error(error);
          document.getElementById("contact-table-body").innerHTML =
            '<tr><td colspan="4" class="empty-line">' + escapeHtml(labels.structure_load_failed) + escapeHtml(error && error.message ? error.message : String(error)) + "</td></tr>";
        });
      }

      loadStructure();
    </script>
  </body>
</html>
"""
    return (
        template.replace("__PAYLOAD_JSON__", payload_json)
        .replace("__HEIGHT_PX__", str(height_px))
        .replace("__NGL_CDN__", NGL_CDN)
    )


def render_structure_viewer(payload: Dict[str, object], height_px: int = 760) -> str:
    html_markup = build_structure_viewer_html(payload, height_px=height_px)
    html(html_markup, height=height_px, scrolling=True)
    return html_markup

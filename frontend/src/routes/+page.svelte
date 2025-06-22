<script lang="ts">
	let modelType = $state('');
	let file: FileList | undefined = $state();
	let target: string = $state('');
	let allCols: string[] = $state([]);
	let featureCols: string[] = $state([]);

	$inspect(featureCols, allCols, target, file, modelType);

	async function toArray() {
		if (!file) {
			return;
		}
		const readedCsv = (await file[0].text()).trim();
		const linhas = readedCsv.split('\n');
		allCols = linhas[0].split(',');
	}

	async function enviarForm() {
		const formData = new FormData();

		if (!file) {
			console.log('Nenhum arquivo');
			return;
		}

		formData.append('file', file[0]);
		formData.append('target', target);
		formData.append('columns', JSON.stringify(featureCols));

		const resposta = await fetch('http://127.0.0.1:5000/gerar', {
			method: 'POST',
			body: formData
		});

		if (resposta.ok == true) {
			console.log(await resposta.text());
		} else {
			console.log('error');
		}
	}

	function addToTarget(col: string) {
		if (featureCols.includes(col)) {
			return;
		}
		target = col;
	}

	function addToFeatures(col: string) {
		if (col === target) {
			return;
		} else if (featureCols.includes(col)) {
			featureCols = featureCols.filter((item) => item !== col);
		} else {
			featureCols.push(col);
		}
		featureCols = featureCols;
	}
</script>

<section class="mt-16">
	<div class="grid grid-cols-2">
		<div>
			<h1>Como utilizar</h1>
			<p class="mt-4">1. Faça o upload da sua base de dados.</p>
			<p>2. Selecione um Target binário. (ex: Sim/Não)</p>
			<p>3. Selecione Features categóricas.</p>
			<p>Obs: Pode ser nescessário tratar dados não catégorico para funcionamento do modelo.</p>
		</div>
		<div>
			<h1>Sobre</h1>
			<p class="mt-4">
				O XGBoost é um dos mais robustos e populares algoritmos de gradient boosting, amplamente
				utilizado em competições e na indústria pela sua precisão e flexibilidade. A abordagem
				utilizada neste notebook também envolveu um pipeline de pré-processamento similar. No
				entanto, para tratar o desbalanceamento de classes, em vez de criar dados novos, foi
				utilizado o parâmetro interno do XGBoost, scale_pos_weight.
			</p>
		</div>
	</div>
	<div>
		<div class="mt-5 flex flex-col gap-4 rounded-xl">
			<div>
				<h1>Upload Data</h1>
				<div class="grid rounded-xl px-2 py-3">
					<h1>*Certifique-se que o seus dados estão no formato .csv</h1>
					<label
						for="button"
						class="cursor-pointer rounded-xl bg-indigo-600 px-4 py-2 text-center text-white transition-colors hover:bg-indigo-500"
						>Upload CSV file</label
					>
					<input
						class="hidden"
						id="button"
						type="file"
						bind:files={file}
						onchange={toArray}
						accept=".csv"
					/>
				</div>
			</div>
			<div class="rounded-xl px-2 py-3">
				{#if target}
					{target}
				{:else}
					<h1>Selecione Target</h1>
				{/if}
			</div>

			{#if allCols}
				<div class="grid grid-cols-2 gap-2">
					{#each allCols as col}
						<button
							class="overflow-x-hidden rounded-md px-2 text-start {col === target
								? 'bg-indigo-500 text-white'
								: 'bg-gray-300'}"
							onclick={() => addToTarget(col)}>{col}</button
						>
					{/each}
				</div>
			{/if}
			<div class="rounded-xl px-2 py-3">
				{#if featureCols.length > 0}
					{featureCols.join(', ')}
				{:else}
					<h1>Selecione Features</h1>
				{/if}
			</div>

			{#if allCols}
				<div class="grid grid-cols-2 gap-2">
					{#each allCols as col}
						<button
							class=" overflow-x-hidden rounded-md {featureCols.includes(col)
								? 'bg-indigo-500 text-white'
								: 'bg-gray-300'}"
							onclick={() => addToFeatures(col)}>{col}</button
						>
					{/each}
				</div>
			{/if}

			<div>
				<h1>Modelos disponiveis</h1>
				<div class="flex justify-center gap-5 rounded-xl bg-gray-300 px-2 py-3">
					<!-- <h1 class="font-semibold">LightGBM</h1>
					<input type="radio" bind:group={modelType} value="LightGBM" /> -->
					<h1 class="font-semibold">XGBoost</h1>
					<input type="radio" bind:group={modelType} value="XGBoost" />
				</div>
			</div>

			<div
				class="flex cursor-pointer justify-center rounded-xl bg-indigo-600 px-2 py-3 text-white transition-all hover:bg-indigo-500"
			>
				<button class="cursor-pointer" onclick={enviarForm}>Testar Modelo</button>
			</div>
		</div>
	</div>
</section>

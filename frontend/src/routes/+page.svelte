<script lang="ts">
	let modelType = $state('')
	let file: FileList | undefined = $state()
	let target = $state('')

	async function enviarForm() {
		const formData = new FormData();

        if (!file){
            console.log("Nenhum arquivo")
            return
        }

		formData.append('file', file[0]);
		formData.append('target', target);
        formData.append('model', modelType)

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
</script>

<nav class="flex grid-cols-2 justify-between">
	<h1>Modelo Machine Learning</h1>
	<h1>Ciências de Dados PUC MG</h1>
</nav>

<section class="grid grid-cols-2">
	<div>
		<h1>Modelos Utilizados</h1>

		<div class="">
			<div>
				<h1>LightGBM</h1>
				<p>
					O LightGBM é um algoritmo de aprendizado de máquina baseado em árvores de decisão,
					reconhecido por sua alta velocidade e eficiência. No contexto do problema, ele foi
					implementado dentro de um pipeline que primeiro pré-processa os dados (padronizando
					valores numéricos e codificando variáveis categóricas) e depois utiliza a técnica SMOTE
					para criar dados sintéticos da classe minoritária (profissionais "insatisfeitos")
				</p>
			</div>

			<div>
				<h1>XGBoost</h1>
				<p>
					O XGBoost é um dos mais robustos e populares algoritmos de gradient boosting, amplamente
					utilizado em competições e na indústria pela sua precisão e flexibilidade. A abordagem
					utilizada neste notebook também envolveu um pipeline de pré-processamento similar. No
					entanto, para tratar o desbalanceamento de classes, em vez de criar dados novos, foi
					utilizado o parâmetro interno do XGBoost, scale_pos_weight.
				</p>
			</div>
		</div>
	</div>

	<div class="flex flex-col gap-4 p-5">
		<input
			class="rounded-xl bg-gray-300 px-2 py-3"
			bind:value={target}
			type="text"
			placeholder="Digite o nome do target"
		/>
		<div class="rounded-xl bg-gray-300 px-2 py-3">
			<input type="file" bind:files={file} accept=".csv" />
			<h1>*Certifique-se que o seus dados estão no formato .csv</h1>
		</div>

		<div class="flex justify-center gap-5 rounded-xl bg-gray-300 px-2 py-3">
			<h1 class="font-semibold">LightGBM</h1>
			<input type="radio" bind:group={modelType} value="LightGBM" />
			<h1 class="font-semibold">XGBoost</h1>
			<input type="radio" bind:group={modelType} value="XGBoost" />
		</div>

		<div class="flex justify-center rounded-xl bg-gray-500 px-2 py-3">
			<button onclick={enviarForm}>Testar Modelo</button>
		</div>
	</div>
</section>
